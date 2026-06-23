# ⚠️ 必须在所有 import 之前
import os
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
from scipy.sparse import coo_matrix, linalg
from scipy.spatial import KDTree
import matplotlib
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib import cm
import subprocess
import glob
import shutil
import re
import csv

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from lasso.dyna import D3plot
except ImportError:
    print("❌ 错误: 未找到 lasso 库。请运行 'pip install lasso-python' 安装。")
    exit()

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator
    prange = range 

try:
    import pypardiso
    PYPARDISO_AVAILABLE = True
except ImportError:
    PYPARDISO_AVAILABLE = False

# ==========================================
# 📝 全局日志捕获类
# ==========================================
class Logger(object):
    def __init__(self, filename="Optimization_Console_Logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# 🚀 提速核心：Numba 原生循环矩阵乘法辅助函数
# ==========================================
@njit(cache=True, fastmath=True)
def mat_mul_2d(A, B):
    I, K = A.shape
    K_B, J = B.shape
    C = np.zeros((I, J))
    for i in range(I):
        for j in range(J):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]
    return C

@njit(cache=True, fastmath=True)
def mat_mul_1d(A, b):
    I, K = A.shape
    c = np.zeros(I)
    for i in range(I):
        for k in range(K):
            c[i] += A[i, k] * b[k]
    return c

# ==========================================
# 1. 🟢 用户全局配置区域
# ==========================================
k_file_path = r"E:\GYZ\20260120-ESL\20260517-try2Dmodel-initial06\comp.k"
work_dir = os.path.dirname(k_file_path)
matsum_path = os.path.join(work_dir, "baseline_run", "matsum") 

DESIGN_PIDS = [1]  
NON_DESIGN_PIDS = []  

EXPANDED_TARGET_PIDS = []
for pid in DESIGN_PIDS:
    EXPANDED_TARGET_PIDS.extend([pid, pid*100+2, pid*100+4, pid*100+6, pid*100+8, pid*100+10])

DYNA_EXE_PATH = r"D:\software\LS_Dyna_R11\R11\program\ls-dyna_smp_d_R11_0_winx64_ifort131.exe"

BRIDGE_VOL_RATIO = 0.60

MAX_MACRO_ITERATIONS = 10      
MACRO_TOLERANCE = 0.005        
INNER_MAX_ITERATIONS = 20     
TARGET_MIN_PCT = 0.55   
TARGET_MAX_PCT = 0.95   

BESO_MAX_ITER = 100           
BESO_ER = 0.02          
BESO_TAU = 0.005              

ENABLE_REBOUND_TRUNCATION = True  
IMPACTOR_NODE_ID = 132725        
IMPACT_DIRECTION = 1              
STEP_INTERVAL = 2   
SCALE_FACTOR = 1.0  

DYNA_INITIAL_DENSITY = 1.0    
SIMP_INITIAL_DENSITY = 1.0 
PENALTY_POWER = 3           
FILTER_RADIUS_RATIO = 6.0     
MOVE_LIMIT = 0.2        
SIMP_OBJ_TOLERANCE = 0.01    

BETA_START = 1.0              
BETA_END = 5.0                 

SYMM_X, SYMM_Y, SYMM_Z = True, False, False  
OUTPUT_SUMMARY_XY, OUTPUT_SUMMARY_XZ, OUTPUT_SUMMARY_YZ = True, False, False     

left_nodes = [116021] + list(range(132467, 132526)) + [116236] 
right_nodes = [116121] + list(range(132009, 132068)) + [116136]
FIXED_NODE_IDS = list(set(left_nodes + right_nodes))
LOAD_NODE_IDS = [] 

E_modulus = 1000
nu = 0.3
MANUAL_THICKNESS = 1.0
MANUAL_DENSITY = 7.85e-9

STAGE_CONSTRAINTS = [(20.0, 70.0)]
GLOBAL_DISP_LIMIT = STAGE_CONSTRAINTS[-1][0]  

CONTACT_ID = 1     
CONTACT_TYPE = 'slave'       
CONTACT_AXIS = 'y'            

# ==========================================
# 2. 工具函数：解析与基础计算
# ==========================================
def build_symmetry_groups(centers, dists_all):
    sym_groups = []
    if not (SYMM_X or SYMM_Y or SYMM_Z): return sym_groups
        
    min_c, max_c = np.min(centers, axis=0), np.max(centers, axis=0)
    mid_c = (max_c + min_c) / 2.0
    visited = np.zeros(len(centers), dtype=bool)
    avg_dist = np.mean(dists_all) if len(dists_all) > 0 else 1.0
    
    tree = KDTree(centers)
    
    for i, center in enumerate(centers):
        if visited[i]: continue
            
        points = [center]
        if SYMM_X: points.extend([[2*mid_c[0]-p[0], p[1], p[2]] for p in points])
        if SYMM_Y: points.extend([[p[0], 2*mid_c[1]-p[1], p[2]] for p in points])
        if SYMM_Z: points.extend([[p[0], p[1], 2*mid_c[2]-p[2]] for p in points])
            
        group = []
        for p in points:
            dist, idx = tree.query(p)
            if dist < avg_dist * 0.5:
                if not visited[idx]: 
                    group.append(idx)
                    visited[idx] = True
                elif idx == i and idx not in group: 
                    group.append(idx)
                    visited[idx] = True
    
        if len(group) > 1: sym_groups.append(group)
            
    return sym_groups

def extract_matsum_centroids(matsum_filepath, target_pids):
    if not os.path.exists(matsum_filepath):
        return {pid: 0.5 for pid in target_pids}, {pid: 'mid' for pid in target_pids}
    ie_data = {pid: {'time': [], 'ie': []} for pid in target_pids}
    current_time = 0.0
   
    with open(matsum_filepath, 'r', errors='ignore') as f:
        for line in f:
            clean_line = line.lower().replace('=', ' = ')
            parts = clean_line.split()
            if not parts: continue
            if 'time' in parts:
                try:
                    t_idx = parts.index('time')
                    if t_idx + 2 < len(parts) and parts[t_idx + 1] == '=': current_time = float(parts[t_idx + 2])
                except: pass
            elif 'mat.# ' in parts and 'inten' in parts:
                try:
                    mat_idx = parts.index('mat.#')
                    inten_idx = parts.index('inten')
                    pid = int(parts[mat_idx + 2])
                    if pid in target_pids:
                        ie = float(parts[inten_idx + 2])
                        ie_data[pid]['time'].append(current_time)
                        ie_data[pid]['ie'].append(ie)
                except: pass
             
    centroids, pid_roles = {}, {}
    for pid in target_pids:
        t_arr, ie_arr = np.array(ie_data[pid]['time']), np.array(ie_data[pid]['ie'])
        if len(t_arr) < 2 or np.max(ie_arr) < 1e-5: 
            centroids[pid] = 0.0
            continue
        die = np.diff(ie_arr)
        die[die < 0] = 0 
        t_mid = 0.5 * (t_arr[1:] + t_arr[:-1])
        total_ie = np.sum(die)
        centroids[pid] = np.sum(t_mid * die) / total_ie if total_ie > 1e-8 else 0.0
            
    c_values = list(centroids.values())
    min_c, max_c = min(c_values), max(c_values)
    assigned_targets = {}
    
    if max_c - min_c < 1e-5: return {pid: 0.6 for pid in target_pids}, {pid: 'mid' for pid in target_pids}
    
    for pid, c_t in centroids.items():
        norm_c = (c_t - min_c) / (max_c - min_c)
        assigned_targets[pid] = np.round(TARGET_MIN_PCT + norm_c * (TARGET_MAX_PCT - TARGET_MIN_PCT), 2)
        pid_roles[pid] = 'front' if norm_c < 0.33 else ('back' if norm_c > 0.66 else 'mid')
            
    return assigned_targets, pid_roles

def run_dyna_safe(k_file, work_dir, log_file=None, timeout_sec=None):
    t_start = time.perf_counter()
    env = os.environ.copy()
    for k in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_CORETYPE']:
        env.pop(k, None)
    cmd = [DYNA_EXE_PATH, f"I={os.path.basename(k_file)}", "NCPU=12"]
    try:
        out_dest = subprocess.DEVNULL
        if log_file:
            with open(os.path.join(work_dir, log_file), "w") as f:
                subprocess.run(cmd, cwd=work_dir, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout_sec)
        else:
            subprocess.run(cmd, cwd=work_dir, stdout=out_dest, stderr=out_dest, env=env, timeout=timeout_sec)
        return True, (time.perf_counter() - t_start)
    except Exception as e: 
        print(f"   ❌ DYNA 运行异常: {e}")
        return False, 0.0

def calculate_initial_masses(elements, nodes_dict, element_pids):
    initial_masses = {pid: 0.0 for pid in np.unique(element_pids)}
    for i, elem in enumerate(elements):
        pid = element_pids[i]
        try:
            p1, p2 = np.array(nodes_dict[elem[0]]), np.array(nodes_dict[elem[1]])
            p3, p4 = np.array(nodes_dict[elem[2]]), np.array(nodes_dict[elem[3]])
            area = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p2))
            initial_masses[pid] += area * MANUAL_THICKNESS * MANUAL_DENSITY
        except: pass
    return initial_masses

def extract_fd_and_ea(work_dir, node_id, dir_idx, contact_id, c_type, c_axis):
    t_nodout, d_raw, t_rcforc, f_raw = [], [], [], []
    current_time, last_disp_time, last_force_time = 0.0, -1.0, -1.0  
    
    nodout_path = os.path.join(work_dir, "nodout")
    if os.path.exists(nodout_path):
        with open(nodout_path, 'r', errors='ignore') as f:
            for line in f:
                line_lower = line.lower()
                if '( at time ' in line_lower:
                    try: current_time = float(line_lower.split('( at time ')[1].split(')')[0].strip())
                    except: pass
                else:
                    parts = re.sub(r'(?<![Ee])-', ' -', line.strip()).split()
                    if len(parts) >= 4 and parts[0] == str(node_id) and current_time > last_disp_time:
                        t_nodout.append(current_time)
                        d_raw.append(float(parts[dir_idx + 1]))
                        last_disp_time = current_time

    rcforc_path = os.path.join(work_dir, "rcforc")
    if os.path.exists(rcforc_path):
        target_type, target_axis = c_type.lower(), c_axis.lower()
        with open(rcforc_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.strip().lower().split()
                if len(parts) > 1 and parts[0] == target_type and parts[1] == str(contact_id):
                    try:
                        if 'time' in parts and target_axis in parts:
                            t_val = float(parts[parts.index('time') + 1])
                            f_val = abs(float(parts[parts.index(target_axis) + 1])) 
                            if t_val > last_force_time:
                                t_rcforc.append(t_val)
                                f_raw.append(f_val)
                            last_force_time = t_val
                    except: pass

    num_stages = len(STAGE_CONSTRAINTS)
    if not t_nodout or not t_rcforc: return [0.0]*num_stages, 0.0, 0.0, np.array([]), np.array([])
        
    t_nodout, d_raw, t_rcforc, f_raw = np.array(t_nodout), np.array(d_raw), np.array(t_rcforc), np.array(f_raw)
    min_d_idx = np.argmin(d_raw)
    t_max_def = t_nodout[min_d_idx]
    t_c, d_c = t_nodout[:min_d_idx + 1], -d_raw[:min_d_idx + 1]
    
    if len(d_c) > 0: d_c = d_c - d_c[0]
    max_disp_val = d_c[-1] if len(d_c) > 0 else 0.0
    valid_f_mask = t_rcforc <= t_max_def
    
    if not np.any(valid_f_mask): return [0.0]*num_stages, max_disp_val, 0.0, d_c, np.zeros_like(d_c)
        
    f_c = np.interp(t_c, t_rcforc, f_raw)
    EA = np.trapz(f_c, d_c)
    
    stage_forces = []
    prev_d = 0.0
    
    for (d_end, _) in STAGE_CONSTRAINTS:
        mask = (d_c > prev_d) & (d_c <= d_end)
        stage_forces.append(np.max(f_c[mask]) if np.any(mask) else 0.0)
        prev_d = d_end
  
    return stage_forces, max_disp_val, EA, d_c, f_c

def save_fd_curve_csv(d_array, f_array, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Displacement(mm),Force(N)\n")
        for d, force in zip(d_array, f_array): f.write(f"{d:.6f},{force:.6f}\n")

class DynaPartReader:
    def __init__(self, filepath, target_pids):
        self.filepath = filepath
        self.target_pids = set(target_pids)
        self.all_nodes = {}
        self.part_elements, self.part_element_ids, self.part_element_pids = [], [], []
        self.part_node_ids = set()

    def read(self):
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f: 
            lines = f.readlines()
            
        current_keyword = None
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("*"): 
                current_keyword = line_str.split()[0].upper()
                continue
            if line_str.startswith("$") or not line_str: continue
            vals = line_str.replace(',', ' ').split()
            if not vals: continue
            try:
                if current_keyword == "*NODE" and len(vals) >= 4:
                    self.all_nodes[int(vals[0])] = [float(vals[1]), float(vals[2]), float(vals[3])]
                elif current_keyword.startswith("*ELEMENT_SHELL") and len(vals) >= 6:
                    pid = int(vals[1])
                    if pid in self.target_pids:
                        eid = int(vals[0])
                        n = [int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5])]
                        self.part_elements.append(n)
                        self.part_element_ids.append(eid)
                        self.part_element_pids.append(pid)
                        self.part_node_ids.update(n)
            except: pass
                
        final_nodes = {nid: self.all_nodes[nid] for nid in self.part_node_ids if nid in self.all_nodes}
        return final_nodes, self.part_elements, self.part_element_ids, np.array(self.part_element_pids)

class D3PlotLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.d3, self.n_steps, self.d3_map = None, 0, None
        
    def load_file(self, load_energy=False):
        filters = ['node_displacement', 'element_shell_internal_energy'] if load_energy else ['node_displacement']
        self.d3 = D3plot(self.filepath, state_array_filter=filters)
        self.node_ids = self.d3.arrays['node_ids']
        self.all_coords = self.d3.arrays['node_displacement'] 
        self.n_steps = self.all_coords.shape[0]
        self.d3_map = {nid: i for i, nid in enumerate(self.node_ids)}

    def find_rebound_step(self, impactor_node_id, direction_idx):
        if self.all_coords is None: self.load_file()
        if impactor_node_id not in self.d3_map: return self.n_steps - 1
            
        node_idx = self.d3_map[impactor_node_id]
        step_velocities = np.diff(self.all_coords[:, node_idx, direction_idx]) 
        
        initial_dir = next((np.sign(v) for v in step_velocities if abs(v) > 1e-5), 0)
        if initial_dir == 0: return self.n_steps - 1
            
        for i, v in enumerate(step_velocities):
            if abs(v) > 1e-5 and np.sign(v) != initial_dir: 
                return min(i + 2, self.n_steps - 1)
        return self.n_steps - 1

    def get_diff_vector(self, step_end, step_start, node_id_map):
        if self.all_coords is None: self.load_file()
        disp_field = self.all_coords[step_end] - self.all_coords[step_start]
        U_vec = np.zeros(len(node_id_map) * 6)
        for nid, matrix_idx in node_id_map.items():
            if nid in self.d3_map: 
                U_vec[matrix_idx*6:matrix_idx*6+3] = disp_field[self.d3_map[nid]] * SCALE_FACTOR
        return U_vec

    def get_total_displacement(self, step, node_id_map):
        if self.all_coords is None: self.load_file()
        disp_field = self.all_coords[step]
        U_vec = np.zeros(len(node_id_map) * 6)
        for nid, matrix_idx in node_id_map.items():
            if nid in self.d3_map: 
                U_vec[matrix_idx*6:matrix_idx*6+3] = disp_field[self.d3_map[nid]]
        return U_vec

# ==========================================
# 3. ⚡ Numba 高性能计算内核 (DiESL) - 极限降维与零分配版
# ==========================================
@njit(cache=True, fastmath=True)
def compute_ke_numba(ele_coords, E, nu, t, density, penalty):
    factor = max(density, 1e-3) ** penalty
    fac = E / (1.0 - nu**2)
    D_val = (E * t**3) / (12.0 * (1.0 - nu**2))
    G = E / (2.0 * (1.0 + nu))
    
    Dm = np.zeros((3, 3))
    Dm[0,0] = 1.0; Dm[0,1] = nu; Dm[1,0] = nu; Dm[1,1] = 1.0; Dm[2,2] = (1.0-nu)/2.0
    Dm = Dm * (fac * factor * t)
    
    Db = np.zeros((3, 3))
    Db[0,0] = 1.0; Db[0,1] = nu; Db[1,0] = nu; Db[1,1] = 1.0; Db[2,2] = (1.0-nu)/2.0
    Db = Db * (D_val * factor)
    
    kG = (5.0/6.0) * G * factor * t
    Ds = np.zeros((2, 2)); Ds[0,0] = kG; Ds[1,1] = kG
    
    T3 = np.zeros((3, 3))
    v1 = ele_coords[1] - ele_coords[0]
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    n = np.cross(v1, ele_coords[3] - ele_coords[0])
    n = n / (np.linalg.norm(n) + 1e-12)
    T3[0,:] = v1; T3[1,:] = np.cross(n, v1); T3[2,:] = n
    
    xy = np.zeros((4, 2))
    for i in range(4):
        diff = ele_coords[i] - ele_coords[0]
        xy[i, 0] = diff[0]*T3[0,0] + diff[1]*T3[0,1] + diff[2]*T3[0,2]
        xy[i, 1] = diff[0]*T3[1,0] + diff[1]*T3[1,1] + diff[2]*T3[1,2]
        
    K_ele = np.zeros((24, 24))
    gp = 1.0 / np.sqrt(3.0)
    gps = np.array([-gp, gp])
    
    for xi in gps:
        for eta in gps:
            dN_dxi = np.zeros((2, 4))
            dN_dxi[0,:] = np.array([-(1.0-eta),  (1.0-eta),  (1.0+eta), -(1.0+eta)]) * 0.25
            dN_dxi[1,:] = np.array([-(1.0-xi), -(1.0+xi),  (1.0+xi),  (1.0-xi)]) * 0.25
            
            J = mat_mul_2d(dN_dxi, xy)
            detJ = max(abs(J[0,0]*J[1,1] - J[0,1]*J[1,0]), 1e-12)
            
            invJ = np.zeros((2, 2))
            invJ[0,0] = J[1,1]/detJ; invJ[0,1] = -J[0,1]/detJ
            invJ[1,0] = -J[1,0]/detJ; invJ[1,1] = J[0,0]/detJ
            
            dN_dxy = mat_mul_2d(invJ, dN_dxi)
            Bm = np.zeros((3, 24)); Bb = np.zeros((3, 24))
            for i in range(4):
                Bm[0, 6*i] = dN_dxy[0, i]; Bm[1, 6*i+1] = dN_dxy[1, i]
                Bm[2, 6*i] = dN_dxy[1, i]; Bm[2, 6*i+1] = dN_dxy[0, i]
                
                Bb[0, 6*i+4] = dN_dxy[0, i]; Bb[1, 6*i+3] = -dN_dxy[1, i]
                Bb[2, 6*i+4] = dN_dxy[1, i]; Bb[2, 6*i+3] = -dN_dxy[0, i]
            
            K_ele += (mat_mul_2d(Bm.T, mat_mul_2d(Dm, Bm)) + mat_mul_2d(Bb.T, mat_mul_2d(Db, Bb))) * detJ
            
    N_0 = np.array([1.0, 1.0, 1.0, 1.0]) * 0.25
    dN_dxi_0 = np.zeros((2, 4))
    dN_dxi_0[0,:] = np.array([-1.0, 1.0, 1.0, -1.0]) * 0.25
    dN_dxi_0[1,:] = np.array([-1.0, -1.0, 1.0, 1.0]) * 0.25
    
    J_0 = mat_mul_2d(dN_dxi_0, xy) 
    detJ_0 = max(abs(J_0[0,0]*J_0[1,1] - J_0[0,1]*J_0[1,0]), 1e-12)
    
    invJ_0 = np.zeros((2, 2))
    invJ_0[0,0] = J_0[1,1]/detJ_0; invJ_0[0,1] = -J_0[0,1]/detJ_0
    invJ_0[1,0] = -J_0[1,0]/detJ_0; invJ_0[1,1] = J_0[0,0]/detJ_0
    
    dN_dxy_0 = mat_mul_2d(invJ_0, dN_dxi_0)
    Bs = np.zeros((2, 24))
    
    for i in range(4):
        Bs[0, 6*i+2] = dN_dxy_0[0, i]; Bs[0, 6*i+4] = N_0[i]
        Bs[1, 6*i+2] = dN_dxy_0[1, i]; Bs[1, 6*i+3] = -N_0[i]
        
    K_ele += mat_mul_2d(Bs.T, mat_mul_2d(Ds, Bs)) * (4.0 * detJ_0)
    for i in range(4): K_ele[6*i+5, 6*i+5] += E * factor * t * 1e-6
        
    T24 = np.zeros((24, 24))
    for i in range(8): T24[3*i:3*i+3, 3*i:3*i+3] = T3
    return mat_mul_2d(T24.T, mat_mul_2d(K_ele, T24))

@njit(cache=True, parallel=True, fastmath=True)
def precompute_connectivity_numba(elements_idx):
    num_ele = elements_idx.shape[0]
    rows = np.zeros(num_ele * 576, dtype=np.int32)
    cols = np.zeros(num_ele * 576, dtype=np.int32)
    for i in prange(num_ele):
        idxs = elements_idx[i]
        g_idx = np.zeros(24, dtype=np.int32)
        for j in range(4):
            for d in range(6): g_idx[j*6+d] = idxs[j]*6 + d
        start = i * 576
        idx_counter = 0
        for r in range(24):
            for c in range(24):
                rows[start + idx_counter] = g_idx[r]
                cols[start + idx_counter] = g_idx[c]
                idx_counter += 1
    return rows, cols

@njit(cache=True, parallel=True, fastmath=True)
def precompute_base_ke_flat_numba(elements_idx, coords, E, nu, t):
    num_ele = elements_idx.shape[0]
    base_ke_flat = np.zeros((num_ele, 576), dtype=np.float64)
    for i in prange(num_ele):
        idxs = elements_idx[i]
        ele_coords = coords[idxs].copy()
        ke = compute_ke_numba(ele_coords, E, nu, t, 1.0, 1.0)
        for r in range(24):
            for c in range(24):
                base_ke_flat[i, r*24+c] = ke[r, c]
    return base_ke_flat

@njit(cache=True, parallel=True, fastmath=True)
def assemble_data_inplace_numba(base_ke_flat, densities, penalty, out_data):
    num_ele = base_ke_flat.shape[0]
    for i in prange(num_ele): 
        factor = max(densities[i], 1e-3) ** penalty
        start = i * 576
        for j in range(576):
            out_data[start + j] = base_ke_flat[i, j] * factor

@njit(cache=True, parallel=True, fastmath=True)
def compute_simp_sensitivity_flat_numba(elements_idx, base_ke_flat, U_global, densities, penalty):
    num_ele = elements_idx.shape[0]
    dc = np.zeros(num_ele, dtype=np.float64)
    for i in prange(num_ele): 
        idxs = elements_idx[i]
        ue = np.zeros(24, dtype=np.float64)
        for j in range(4):
            for d in range(6): ue[j*6+d] = U_global[idxs[j]*6+d]
        
        strain_energy = 0.0
        for r in range(24):
            row_val = 0.0
            for c in range(24):
                row_val += base_ke_flat[i, r*24+c] * ue[c]
            strain_energy += 0.5 * ue[r] * row_val
            
        dc[i] = penalty * (max(densities[i], 1e-3) ** (penalty - 1.0)) * strain_energy
    return dc  

@njit(cache=True, fastmath=True)
def build_coo_to_csr_map(free_rows, free_cols, indptr, indices):
    N = len(free_rows)
    mapping = np.zeros(N, dtype=np.int32)
    for k in range(N):
        r = free_rows[k]
        c = free_cols[k]
        start = indptr[r]
        end = indptr[r+1]
        for p in range(start, end):
            if indices[p] == c:
                mapping[k] = p
                break
    return mapping

@njit(cache=True, fastmath=True)
def accumulate_csr_data_numba(free_data, mapping, csr_data):
    for k in range(len(free_data)):
        csr_data[mapping[k]] += free_data[k]

@njit(cache=True, fastmath=True)
def apply_symmetry_inplace_numba(field, sym_indices, sym_offsets):
    """【新加速】消除 Python 级 for 循环，底层 C 级速度抹平对称组"""
    n_groups = len(sym_offsets) - 1
    for i in range(n_groups):
        start = sym_offsets[i]
        end = sym_offsets[i+1]
        s = 0.0
        for j in range(start, end):
            s += field[sym_indices[j]]
        mean_val = s / (end - start)
        for j in range(start, end):
            field[sym_indices[j]] = mean_val

@njit(cache=True, parallel=True, fastmath=True)
def heaviside_inplace_numba(x_tilde, beta, eta, out):
    """【新加速】内存原地覆盖的 Heaviside 过滤，完全避免垃圾回收"""
    n = len(x_tilde)
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    for i in prange(n):
        num = np.tanh(beta * eta) + np.tanh(beta * (x_tilde[i] - eta))
        out[i] = num / den

# ==========================================
# 4. DiESL 核心优化器
# ==========================================
from scipy.sparse import csr_matrix

class SRIShellAssembler:
    def __init__(self, nodes_dict, elements_list, E, nu, t, fixed_nodes):
        self.nodes_dict, self.elements = nodes_dict, elements_list
        self.E, self.nu, self.t = E, nu, t
        sorted_node_ids = sorted(self.nodes_dict.keys())
    
        self.node_id_map = {nid: i for i, nid in enumerate(sorted_node_ids)}
        self.n_nodes = len(sorted_node_ids)
        self.n_dof = self.n_nodes * 6 
        
        fixed_dofs = [self.node_id_map[nid]*6+i for nid in fixed_nodes if nid in self.node_id_map for i in range(6)]
        self.free_dofs = np.setdiff1d(np.arange(self.n_dof), fixed_dofs)
        self.n_free = len(self.free_dofs)
        
        self.dof_map = np.full(self.n_dof, -1, dtype=np.int32)
        self.dof_map[self.free_dofs] = np.arange(self.n_free)
        
        self.initial_coords_array = np.array([self.nodes_dict[nid] for nid in sorted_node_ids])
        self.coords_array = self.initial_coords_array.copy()
        self.elements_idx_array = np.array([[self.node_id_map[n] for n in elem] for elem in self.elements], dtype=np.int32)
        
        self.cached_rows, self.cached_cols = precompute_connectivity_numba(self.elements_idx_array)
        self.k_data_buffer = np.zeros(len(self.elements) * 576, dtype=np.float64)
        
        mapped_rows = self.dof_map[self.cached_rows]
        mapped_cols = self.dof_map[self.cached_cols]
        self.valid_mask = (mapped_rows >= 0) & (mapped_cols >= 0)
        self.free_rows = mapped_rows[self.valid_mask]
        self.free_cols = mapped_cols[self.valid_mask]

        print("      🔍 正在初始化零切片 CSR 内存映射字典 (单次耗时)...")
        dummy_data = np.ones(len(self.free_rows), dtype=np.float64)
        dummy_coo = coo_matrix((dummy_data, (self.free_rows, self.free_cols)), shape=(self.n_free, self.n_free))
        self.base_csr = dummy_coo.tocsr() 

        self.coo_to_csr_map = build_coo_to_csr_map(self.free_rows, self.free_cols, self.base_csr.indptr, self.base_csr.indices)
        print("      ✅ 映射字典构建完成！")
        
    def update_coords(self, U_global):
        self.coords_array = self.initial_coords_array.copy()
        for nid, idx in self.node_id_map.items(): 
            self.coords_array[idx] += U_global[idx*6 : idx*6+3]
            
    def precompute_case_bases(self, esl_cases):
        for case in esl_cases:
            case['base_ke_flat'] = precompute_base_ke_flat_numba(self.elements_idx_array, case['coords'], self.E, self.nu, self.t)
            
    def assemble_global(self, densities=None, penalty=3.0):
        if densities is None: densities = np.ones(len(self.elements))
        temp_base_ke_flat = precompute_base_ke_flat_numba(self.elements_idx_array, self.coords_array, self.E, self.nu, self.t)
        assemble_data_inplace_numba(temp_base_ke_flat, densities, penalty, self.k_data_buffer)
        return coo_matrix((self.k_data_buffer, (self.cached_rows, self.cached_cols)), shape=(self.n_dof, self.n_dof)).tocsr()
        
    def assemble_free_csr(self, base_ke_flat, densities=None, penalty=3.0):
        if densities is None: densities = np.ones(len(self.elements))
        assemble_data_inplace_numba(base_ke_flat, densities, penalty, self.k_data_buffer)
        free_data = self.k_data_buffer[self.valid_mask]
        
        self.base_csr.data.fill(0.0)
        accumulate_csr_data_numba(free_data, self.coo_to_csr_map, self.base_csr.data)
        
        return csr_matrix((self.base_csr.data, self.base_csr.indices, self.base_csr.indptr), shape=(self.n_free, self.n_free), copy=False)
        
    def compute_simp_sensitivity_cached(self, base_ke_flat, U_global, densities, penalty=3.0):
        return compute_simp_sensitivity_flat_numba(self.elements_idx_array, base_ke_flat, U_global, densities, penalty)

class RealTimeVisualizer:
    def __init__(self, centers):
        plt.ion() 
        self.fig = plt.figure(figsize=(7, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.centers, self.frames = centers, [] 
        mid = np.mean(centers, axis=0)
        max_range = np.array([centers[:,0].ptp(), centers[:,1].ptp(), centers[:,2].ptp()]).max() / 2.0
        
        self.ax.set(xlim=(mid[0]-max_range, mid[0]+max_range),
                    ylim=(mid[1]-max_range, mid[1]+max_range),
                    zlim=(mid[2]-max_range, mid[2]+max_range),
                    title="Real-time SIMP Evolution")
        self.ax.axis('off')
       
        try: self.ax.set_box_aspect((1, 1, 1))
        except: pass
        
        self.scat = self.ax.scatter(centers[:,0], centers[:,1], centers[:,2], c='blue', cmap='jet', vmin=0, vmax=1, s=15, alpha=0.8)
        self.fig.colorbar(self.scat, ax=self.ax, label="Relative Density", shrink=0.5)
        plt.show()

    def update(self, densities, global_iter, beta):
        try:
            if not plt.fignum_exists(self.fig.number): return 
  
            self.scat.set_array(densities)
            self.ax.set_title(f"Real-time SIMP | Global Iter: {global_iter} | Beta: {beta:.1f}")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            w, h = self.fig.canvas.get_width_height()
            self.frames.append(np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3)))
            plt.pause(0.001)
        except: pass

    def save_animation(self, save_path, fps=5):
        if self.frames:
            print(f"      🎬 正在保存动图 (FPS={fps})...")
            try: 
                imageio.mimsave(save_path, self.frames, duration=int(1000/fps), loop=0)
                print(f"      ✅ 动图已生成: {save_path}")
            except Exception as e: print(f"      ❌ 动图保存失败: {e}")

class SIMP_Optimizer:
    def __init__(self, assembler, target_vols_dict, element_pids, penalty, r_filter_ratio, move_limit, triple_loop_prefix=""):
        self.assembler, self.target_vols, self.element_pids = assembler, target_vols_dict, element_pids
        self.p, self.move, self.prefix = penalty, move_limit, triple_loop_prefix 
        self.global_iter = 0
        self.history_compliance, self.history_volume_fraction, self.esl_cases = [], [], []

        self.centers = np.array([np.mean([assembler.nodes_dict[n] for n in elem], axis=0) for elem in assembler.elements])
        dists_all = np.linalg.norm(self.centers[1:] - self.centers[:-1], axis=1) if len(self.centers) > 1 else np.array([1.0])
        self.r_min = (np.mean(dists_all) if len(dists_all) > 0 else 1.0) * r_filter_ratio
        
        self.tree = KDTree(self.centers)
        neighbor_indices = self.tree.query_ball_point(self.centers, self.r_min)
       
        rows, cols, data = [], [], []
        for i, neighbors in enumerate(neighbor_indices):
            valid = [n for n in neighbors if self.element_pids[n] == self.element_pids[i]]
            if valid:
                weights = np.maximum(0, self.r_min - np.linalg.norm(self.centers[i] - self.centers[valid], axis=1))
                sum_w = np.sum(weights)
                if sum_w > 1e-12: 
                    rows.extend([i]*len(valid)); cols.extend(valid); data.extend(weights / sum_w)
                else: 
                    rows.append(i); cols.append(i); data.append(1.0)
            else: 
                rows.append(i); cols.append(i); data.append(1.0)
                
        self.H = coo_matrix((data, (rows, cols)), shape=(len(self.centers), len(self.centers))).tocsr()
        self.HT = self.H.transpose()

        protected_nodes = set(FIXED_NODE_IDS + LOAD_NODE_IDS)
        self.passive_indices = list(set([i for i, elem in enumerate(assembler.elements) if any(n in protected_nodes for n in elem)]))
        self.sym_groups = build_symmetry_groups(self.centers, dists_all)
        
        flat_sym, sym_offs = [], [0]
        for g in self.sym_groups:
            flat_sym.extend(g)
            sym_offs.append(len(flat_sym))
        self.sym_indices = np.array(flat_sym, dtype=np.int32)
        self.sym_offsets = np.array(sym_offs, dtype=np.int32)

        if PYPARDISO_AVAILABLE:
            self.solver = pypardiso.PyPardisoSolver()
            print(f"      ⚡ 已激活 PyPardiso 求解器实例，开启矩阵符号分解缓存！")

    def apply_symmetry(self, field):
        if not self.sym_groups: return field
        new_field = np.copy(field)
        for group in self.sym_groups: new_field[group] = np.mean(new_field[group])
        return new_field

    def clear_esl_cases(self): self.esl_cases.clear()
    def add_esl_case(self, coords_state, f_esl): self.esl_cases.append({'coords': coords_state, 'f_esl': f_esl})

    def heaviside(self, x_tilde, beta, eta=0.5):
        num = np.tanh(beta * eta) + np.tanh(beta * (x_tilde - eta))
        return num / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))

    def heaviside_grad(self, x_tilde, beta, eta=0.5):
        return beta * (1.0 - np.tanh(beta * (x_tilde - eta))**2) / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))

    def optimality_criteria_update(self, x, dc_dx, dv_dx, beta):
        unique_pids = [pid for pid in np.unique(self.element_pids) if pid not in NON_DESIGN_PIDS]
        num_pids = len(unique_pids)
      
        target_v = {pid: self.target_vols[pid] * np.sum(self.element_pids == pid) for pid in unique_pids}
        dc_dv_ratio = np.maximum(0.0, dc_dx) / np.maximum(1e-10, dv_dx)
        
        l1 = np.zeros(num_pids)
        l2 = np.ones(num_pids) * max(10.0, np.max(dc_dv_ratio) * 3.0) 
        
        x_cand, x_phys = np.zeros_like(x), np.zeros_like(x)
        pid_masks = [self.element_pids == pid for pid in unique_pids]
        l_mid_full = np.empty_like(x)

        while np.max((l2 - l1) / (l1 + l2 + 1e-12)) > 1e-3:
            l_mid_array = 0.5 * (l2 + l1)
            for i, mask in enumerate(pid_masks): 
                l_mid_full[mask] = l_mid_array[i]
            
            np.sqrt(dc_dv_ratio / l_mid_full, out=x_cand)
            x_cand *= x
            np.clip(x_cand, np.maximum(0.001, x - self.move), np.minimum(1.0, x + self.move), out=x_cand)
            x_cand[self.passive_indices] = 1.0
   
            if len(self.sym_offsets) > 1:
                apply_symmetry_inplace_numba(x_cand, self.sym_indices, self.sym_offsets)
    
            x_tilde_tmp = self.H.dot(x_cand) 
            heaviside_inplace_numba(x_tilde_tmp, beta, 0.5, x_phys)
            
            x_phys[self.passive_indices] = 1.0 
            
            for i, pid in enumerate(unique_pids):
                if np.sum(x_phys[pid_masks[i]]) > target_v[pid]: l1[i] = l_mid_array[i]
                else: l2[i] = l_mid_array[i]
                   
        return x_cand, x_phys

    def run_optimization(self, current_x, current_beta, macro_step, global_time_logs, max_iter=20):
        x = np.copy(current_x)  
        N_cases = max(1, len(self.esl_cases))
        base_comp = None 
        x_tilde = self.H.dot(x)
        x_phys = self.heaviside(x_tilde, current_beta)

        if not hasattr(self, 'visualizer'): self.visualizer = RealTimeVisualizer(self.centers)

        n_dof = self.assembler.n_dof
        free_dofs = self.assembler.free_dofs

        self.assembler.precompute_case_bases(self.esl_cases)

        for k in range(max_iter):
             t_local_start = time.perf_counter()
             self.global_iter += 1
             total_comp, total_dc_dbar = 0.0, np.zeros(len(x))
            
             for case in self.esl_cases:
                 t_asm_start = time.perf_counter()
                 K_free = self.assembler.assemble_free_csr(case['base_ke_flat'], densities=x_phys, penalty=self.p)
                 F_free = case['f_esl'][free_dofs]
           
                 t_asm_end = time.perf_counter()
                 
                 if PYPARDISO_AVAILABLE:
                     U_f = self.solver.solve(K_free, F_free)
                 else:
                     U_f = linalg.spsolve(K_free, F_free)
                 t_sol_end = time.perf_counter()

                 U_iter = np.zeros(n_dof)
                 U_iter[free_dofs] = U_f
                 
                 total_comp += 0.5 * case['f_esl'].dot(U_iter)
                 total_dc_dbar += self.assembler.compute_simp_sensitivity_cached(case['base_ke_flat'], U_iter, densities=x_phys, penalty=self.p)
                 t_sens_end = time.perf_counter()
                 print(f"      [耗时监控] 组装: {t_asm_end-t_asm_start:.3f}s | 求解: {t_sol_end-t_asm_end:.3f}s | 敏度: {t_sens_end-t_sol_end:.3f}s")

             base_comp = total_comp if k == 0 and total_comp > 1e-12 else (base_comp or 1e-12)
             norm_comp = (total_comp / base_comp) * N_cases
             self.history_compliance.append(norm_comp)
             
             if len(self.history_compliance) >= 2:
                 obj_change = abs(self.history_compliance[-1] - self.history_compliance[-2]) / (self.history_compliance[-2] + 1e-12)
             else:
                 obj_change = 1.0
            
             dbar_dtilde = self.heaviside_grad(x_tilde, current_beta)
             dc_dx = self.apply_symmetry(self.HT.dot(total_dc_dbar * dbar_dtilde)) 
             dv_dx = self.apply_symmetry(self.HT.dot(dbar_dtilde))

             x_new, x_phys_new = self.optimality_criteria_update(x, dc_dx, dv_dx, current_beta)
             vol_curr = np.mean(x_phys_new)
             self.history_volume_fraction.append(vol_curr)
             change = np.max(np.abs(x_new - x))
      
             t_local_end = time.perf_counter()
             local_time = t_local_end - t_local_start

             print(f"      [Global Iter {self.global_iter:<3} | Beta: {current_beta:.1f}] Local {k+1:<2} | Norm Comp: {norm_comp:.4f} | Vol: {vol_curr:.3f} | Change: {change:.4f} | Obj Error: {obj_change:.2e}")
             print(f"      ⏱️ [耗时监控] 本次 SIMP 迭代 (Local {k+1}) 总计耗时: {local_time:.3f}s")
             
             global_time_logs.append({
                 "Phase": "DiESL_SIMP",
                 "Macro_Iter": macro_step,
                 "DYNA_Time_s": "-",
                 "SIMP_Local_Iter": k + 1,
                 "SIMP_Time_s": round(local_time, 3),
                 "BESO_Iter": "-",
                 "BESO_Time_s": "-"
             })

             self.visualizer.update(x_phys_new, self.global_iter, current_beta)

             x, x_tilde, x_phys = x_new, self.H.dot(x_new), x_phys_new
             
             if (change < 0.01 or obj_change < SIMP_OBJ_TOLERANCE) and k > 5: 
                 print(f"      🎉 内部 SIMP 迭代收敛! (Change: {change:.4f}, Obj Error: {obj_change:.2e})")
                 break
       
        self.plot_convergence()
        if hasattr(self, 'visualizer'): self.visualizer.save_animation(os.path.join(os.path.dirname(k_file_path), f"{self.prefix}evolution.gif"), fps=5) 
            
        return x, x_tilde, x_phys

    def plot_convergence(self):
        iters = range(1, len(self.history_compliance) + 1)
        try:
            with open(os.path.join(os.path.dirname(k_file_path), f"{self.prefix}convergence_data.csv"), "w", encoding="utf-8") as f:
                f.write("Global_Iteration,Normalized_Compliance,Volume_Fraction\n")
                for i in range(len(iters)): 
                     f.write(f"{iters[i]},{self.history_compliance[i]:.6f},{self.history_volume_fraction[i]:.6f}\n")
        except: pass
           
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Global Iteration Step')
        ax1.set_ylabel('Normalized Objective $\Omega$', color='tab:blue', fontsize=12)
        ax1.plot(iters, self.history_compliance, color='tab:blue', lw=2, marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume Fraction', color='tab:red', fontsize=12)
        ax2.plot(iters, self.history_volume_fraction, color='tab:red', lw=1.5, ls='--', marker='x')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')
       
        plt.title('ESL Macro-Loop Topology Optimization Convergence')
        fig.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(k_file_path), f"{self.prefix}convergence.png"))
        plt.close(fig)

class KFileUpdater:
    def __init__(self, src_path, dst_path, target_pids):
        self.src, self.dst, self.target_pids = src_path, dst_path, set(target_pids)

    def update(self, elements, element_ids, densities, element_pids, crisp_thresholds=None):
        with open(self.src, 'r', encoding='utf-8', errors='ignore') as fin, open(self.dst, 'w', encoding='utf-8') as fout:
            in_element_block = False
            for line in fin:
                line_strip = line.strip().upper()
                if line_strip.startswith("*"):
                    in_element_block = False
                    if line_strip == "*END": 
                        self._write_new_elements(fout, elements, element_ids, densities, element_pids, crisp_thresholds)
                        fout.write("*END\n")
                        break
                    if line_strip.startswith("*ELEMENT_SHELL"): 
                        in_element_block = True
                        fout.write(line)
                        continue
                    fout.write(line)
                    continue
                    
                if in_element_block:
                    if line_strip.startswith("$") or not line_strip: 
                        fout.write(line)
                        continue
                    parts = line.replace(',', ' ').split()
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) in self.target_pids: 
                        continue
                    fout.write(line)
                else: 
                    fout.write(line)

    def _write_new_elements(self, fout, elements, element_ids, densities, element_pids, crisp_thresholds):
        fout.write("\n$ === SIMP OPTIMIZED ELEMENTS ===\n*ELEMENT_SHELL\n$      EID       PID      N1      N2      N3      N4\n")
        for i, d in enumerate(densities):
            orig_pid = element_pids[i]
            if orig_pid in NON_DESIGN_PIDS: continue 
  
            if crisp_thresholds is not None:
                 thresh = crisp_thresholds.get(orig_pid, 0.5)
                 if d < thresh - 1e-5: continue
                 new_pid = orig_pid
            else:
                d_clamp = max(0.0, min(1.0, d))
                level = 10 if d_clamp >= 0.8 else (8 if d_clamp >= 0.6 else (6 if d_clamp >= 0.4 else (4 if d_clamp >= 0.2 else 2)))
                new_pid = orig_pid * 100 + level
      
            ns = elements[i]
            fout.write(f"{element_ids[i]:8d}{new_pid:8d}{ns[0]:8d}{ns[1]:8d}{ns[2]:8d}{ns[3]:8d}\n")

def plot_simp_result(nodes_dict, elements_list, densities, prefix_name):
    polys, colors = [], []
    for i, elem in enumerate(elements_list):
        if densities[i] >= 0.0:
            try: 
                polys.append([nodes_dict[n] for n in elem])
                colors.append(densities[i])
            except: pass
                
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    norm = plt.Normalize(vmin=0, vmax=1)
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cm.jet(norm(colors)), edgecolors='none', alpha=0.9))
    
    if polys:
        verts = np.array([v for p in polys for v in p])
        mid = np.mean(verts, axis=0)
        max_range = np.array([verts[:,0].ptp(), verts[:,1].ptp(), verts[:,2].ptp()]).max() / 2.0
        ax.set(xlim=(mid[0]-max_range, mid[0]+max_range), ylim=(mid[1]-max_range, mid[1]+max_range), zlim=(mid[2]-max_range, mid[2]+max_range))
    
    m = cm.ScalarMappable(norm=norm, cmap=cm.jet); m.set_array(colors)
    plt.colorbar(m, ax=ax, label="Relative Density")
    plt.savefig(os.path.join(os.path.dirname(k_file_path), f"{prefix_name}.png"))
    plt.close(fig)

def plot_summary_2d_view(nodes_dict, elements_list, macro_densities_list, save_dir, plane='XY', prefix=""):
    if not macro_densities_list: return
        
    cols = 3
    rows = int(np.ceil(len(macro_densities_list) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes = np.atleast_1d(axes).flatten()
    norm = plt.Normalize(vmin=0, vmax=1)
    
    idx1, idx2 = (0, 1) if plane == 'XY' else ((0, 2) if plane == 'XZ' else (1, 2))
        
    for i, densities in enumerate(macro_densities_list):
        polys_2d, colors = [], []
        for j, elem in enumerate(elements_list):
            if densities[j] >= 0.0:
                try: 
                    polys_2d.append([[nodes_dict[n][idx1], nodes_dict[n][idx2]] for n in elem])
                    colors.append(densities[j])
                except: pass
                    
        axes[i].add_collection(PolyCollection(polys_2d, facecolors=cm.jet(norm(colors)), edgecolors='none', alpha=0.9))
        axes[i].autoscale_view(); axes[i].set_aspect('equal'); axes[i].set_title(f"Iter {i+1}"); axes[i].axis('off')
        
    for i in range(len(macro_densities_list), len(axes)): axes[i].axis('off')
        
    sm = cm.ScalarMappable(norm=norm, cmap=cm.jet); sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='vertical', shrink=0.8, pad=0.02).set_label("Relative Density")
    fig.suptitle(f"Topology Progression ({plane} Plane)", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"{prefix}topology_summary_{plane}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_diesl_double_loop(target_domains, work_dir, run_k_file, nodes, elements, element_ids, element_pids, prefix_name, global_time_logs):
    assembler = SRIShellAssembler(nodes, elements, E_modulus, nu, MANUAL_THICKNESS, FIXED_NODE_IDS)
    optimizer = SIMP_Optimizer(assembler, target_domains, element_pids, PENALTY_POWER, FILTER_RADIUS_RATIO, MOVE_LIMIT, prefix_name)

    print(f"\n   [初始化] 将物理初始密度 {DYNA_INITIAL_DENSITY} 映射到 {prefix_name} 初始模型...")
    current_x = np.ones(len(elements)) * SIMP_INITIAL_DENSITY
    prev_U_vec, macro_densities_history = None, []

    temp_init_k = os.path.join(work_dir, "temp_init.k")
    KFileUpdater(run_k_file, temp_init_k, EXPANDED_TARGET_PIDS).update(elements, element_ids, current_x, element_pids)
    shutil.move(temp_init_k, run_k_file)

    diesl_metrics_csv = os.path.join(work_dir, f"{prefix_name}Timing_Metrics.csv")
    with open(diesl_metrics_csv, 'w', encoding='utf-8') as f:
        f.write("Macro_Iter,DYNA_Time_s,SIMP_Inner_Time_s,Macro_Total_Time_s\n")

    for macro_step in range(MAX_MACRO_ITERATIONS):
        t_macro_start = time.perf_counter() 
        
        current_beta = BETA_START + (BETA_END - BETA_START) * (macro_step / max(1, MAX_MACRO_ITERATIONS - 1))
        print(f"\n   [DiESL Phase 1 {macro_step+1}/{MAX_MACRO_ITERATIONS} | Beta={current_beta:.1f}] 宏观碰撞与载荷重构...")
        
        for f in glob.glob(os.path.join(work_dir, "d3plot*")) + glob.glob(os.path.join(work_dir, "d3dump*")) + [os.path.join(work_dir, "nodout"), os.path.join(work_dir, "rcforc")]:
            try: os.remove(f)
            except: pass

        success, dyna_time = run_dyna_safe(run_k_file, work_dir, f"dyna_phase1_macro_{macro_step+1}.log")

        global_time_logs.append({
            "Phase": "DiESL_Macro_DYNA",
            "Macro_Iter": macro_step + 1,
            "DYNA_Time_s": round(dyna_time, 3),
            "SIMP_Local_Iter": "-",
            "SIMP_Time_s": "-",
            "BESO_Iter": "-",
            "BESO_Time_s": "-"
        })

        F_stages_phase1, D_max_phase1, EA_phase1, _, _ = extract_fd_and_ea(work_dir, IMPACTOR_NODE_ID, IMPACT_DIRECTION, CONTACT_ID, CONTACT_TYPE, CONTACT_AXIS)
        print(f"        📈 碰撞响应: [{', '.join([f'S{i+1}={f:.1f}N' for i, f in enumerate(F_stages_phase1)])}], D={D_max_phase1:.2f}mm | EA={EA_phase1:.2f}")
        print(f"      ⏱️ [耗时统计] LS-DYNA 非线性求解耗时: {dyna_time:.2f} 秒")

        d3_loader = D3PlotLoader(os.path.join(work_dir, "d3plot"))
        try: d3_loader.load_file()
        except: raise RuntimeError("❌ 严重错误: LS-DYNA 未能生成 d3plot！请检查日志。")

        cutoff_step = d3_loader.find_rebound_step(IMPACTOR_NODE_ID, IMPACT_DIRECTION) if ENABLE_REBOUND_TRUNCATION else d3_loader.n_steps - 1
        curr_U_vec = d3_loader.get_total_displacement(max(1, cutoff_step), assembler.node_id_map)
     
        if prev_U_vec is not None and curr_U_vec is not None:
            change = np.linalg.norm(curr_U_vec - prev_U_vec) / (np.linalg.norm(prev_U_vec) + 1e-12)
            print(f"      📊 全局位移场变化率: {change*100:.2f}%")
            if change < MACRO_TOLERANCE: 
                print("      ✅ 宏观物理场收敛，DiESL 结束！")
                break
                
        prev_U_vec = curr_U_vec.copy()

        step_indices = list(range(0, cutoff_step + 1, STEP_INTERVAL))
        if not step_indices or step_indices[-1] != cutoff_step: step_indices.append(cutoff_step)
            
        optimizer.clear_esl_cases()
        temp_x_phys = optimizer.heaviside(optimizer.H.dot(current_x), current_beta)
        
        for i in range(len(step_indices) - 1):
            idx_start, idx_end = step_indices[i], step_indices[i+1]
            assembler.update_coords(d3_loader.get_total_displacement(idx_start, assembler.node_id_map))
            current_deformed_coords = assembler.coords_array.copy()
            
            K_current = assembler.assemble_global(densities=temp_x_phys, penalty=PENALTY_POWER)
            U_diff = d3_loader.get_diff_vector(idx_end, idx_start, assembler.node_id_map)
            
            if U_diff is not None: 
                optimizer.add_esl_case(current_deformed_coords, K_current.dot(U_diff))

        t_simp_start = time.perf_counter()
        current_x, current_x_tilde, current_x_phys = optimizer.run_optimization(current_x, current_beta, macro_step + 1, global_time_logs, max_iter=INNER_MAX_ITERATIONS)
        t_simp_end = time.perf_counter()
        simp_time = t_simp_end - t_simp_start
        print(f"      ⏱️ [耗时统计] 内部 SIMP 迭代计算耗时: {simp_time:.2f} 秒")

        macro_densities_history.append(np.copy(current_x_phys))
        
        temp_k_path = os.path.join(work_dir, "temp_run.k")
        KFileUpdater(run_k_file, temp_k_path, EXPANDED_TARGET_PIDS).update(elements, element_ids, current_x_phys, element_pids)
        shutil.move(temp_k_path, run_k_file)
        plot_simp_result(nodes, elements, current_x_phys, f"{prefix_name}macro_{macro_step+1}")
        
        t_macro_end = time.perf_counter()
        macro_time = t_macro_end - t_macro_start
        with open(diesl_metrics_csv, 'a', encoding='utf-8') as f:
            f.write(f"{macro_step+1},{dyna_time:.2f},{simp_time:.2f},{macro_time:.2f}\n")

    if OUTPUT_SUMMARY_XY: plot_summary_2d_view(nodes, elements, macro_densities_history, work_dir, 'XY', prefix_name)
    if OUTPUT_SUMMARY_XZ: plot_summary_2d_view(nodes, elements, macro_densities_history, work_dir, 'XZ', prefix_name)
    if OUTPUT_SUMMARY_YZ: plot_summary_2d_view(nodes, elements, macro_densities_history, work_dir, 'YZ', prefix_name)

    plt.ioff(); plt.close('all')
    return current_x_phys

# ==========================================
# 5. 🌟 BESO 硬杀伤算法与重塑模块
# ==========================================
def apply_beso_hardkill(initial_k, new_k, void_ids, design_pids):
    modified_solid, modified_void = 0, 0
    in_shell = False 
    
    with open(initial_k, 'r') as f_in, open(new_k, 'w') as f_out:
        for line in f_in:
            stripped = line.strip()
            if stripped.startswith("*ELEMENT_SHELL"): 
                in_shell = True
                f_out.write(line)
                continue
            elif stripped.startswith("*"): 
                in_shell = False
                f_out.write(line)
                continue
             
            if in_shell and not stripped.startswith("$") and stripped and len(line) >= 16:
                try:
                    eid, current_pid = int(line[0:8]), int(line[8:16])
                    if current_pid in design_pids:
                        if eid in void_ids: modified_void += 1
                        else: f_out.write(line); modified_solid += 1
                    else: f_out.write(line) 
                except ValueError: f_out.write(line)
            else: f_out.write(line)
                
    return modified_solid, modified_void

def extract_beso_voids(d3plot_path, target_vols, element_ids, element_pids, centers, vols, r_min, history_sens=None, impactor_node_id=132725, impact_dir=1, enable_rebound=True, sym_groups=None):
    print(f"      📖 读取 D3PLOT 提取最大位移时刻的塑性内能...")
    
    d3_loader = D3PlotLoader(d3plot_path)
    d3_loader.load_file(load_energy=True) 
    cutoff_step = d3_loader.find_rebound_step(impactor_node_id, impact_dir) if enable_rebound else d3_loader.n_steps - 1
    
    if 'element_shell_internal_energy' not in d3_loader.d3.arrays.keys(): raise KeyError("找不到壳单元内能数据，请检查 K 文件 ENGFLG 设置！")

    energy_map = {eid: e for eid, e in zip(d3_loader.d3.arrays['element_shell_ids'], d3_loader.d3.arrays['element_shell_internal_energy'][cutoff_step])}
    
    total_V, total_E = np.sum(vols), np.sum(list(energy_map.values())) + 1e-12 
    sens = np.array([(vols[i] / total_V) - (energy_map.get(eid, 0.0) / total_E) for i, eid in enumerate(element_ids)])
      
    print(f"      🌊 执行 BESO 空间滤波与时间历史平滑 (为虚单元假想内能)...")
    tree = KDTree(centers)
    filtered_sens = np.zeros_like(sens)
    
    for i, neighbors in enumerate(tree.query_ball_point(centers, r_min)):
        weights = np.maximum(0, r_min - np.linalg.norm(centers[i] - centers[neighbors], axis=1))
        sum_w = np.sum(weights)
        filtered_sens[i] = np.sum(weights * sens[neighbors]) / sum_w if sum_w > 1e-12 else sens[i]
        
    if history_sens is not None: filtered_sens = (filtered_sens + history_sens) / 2.0
    
    if sym_groups:
        for group in sym_groups: filtered_sens[group] = np.mean(filtered_sens[group])
            
    void_ids = set()
    for pid in DESIGN_PIDS:
        pid_indices = np.where(element_pids == pid)[0]
        n_keep = int(len(pid_indices) * target_vols[pid])
        void_ids.update(element_ids[pid_indices[idx]] for idx in np.argsort(filtered_sens[pid_indices])[n_keep:])
      
    return void_ids, filtered_sens.copy()

# ==========================================
# 6. 🚀 串联主程序：DiESL -> 60% 桥接 -> BESO
# ==========================================
def main():
    # --- 接管全局控制台输出 ---
    log_file_path = os.path.join(work_dir, "Optimization_Console_Logs.txt")
    sys.stdout = Logger(log_file_path)
    
    # --- 统一的时间记录列表 ---
    global_time_logs = []
    
    t_global_start = time.perf_counter() 

    run_k_file = os.path.join(work_dir, "run_optim_phase1.k")
        
    print("\n" + "🔥"*35)
    print(" 🚀 启动结构寻优：两阶段串联架构 (DiESL + BESO [硬杀伤版])")
    print(" 📊 Phase 1: 寻找最优等效静载骨架")
    print(f" 🔗 Bridge : 按照 {BRIDGE_VOL_RATIO:.0%} 进行绝对实/虚截断 (直接物理删除虚单元)")
    print(" 📊 Phase 2: 基于空间外推的 BESO 极限吸能重塑 (无虚单元纯净计算)")
    print("🔥"*35)

    try: 
        nodes, elements, element_ids, element_pids = DynaPartReader(k_file_path, EXPANDED_TARGET_PIDS + NON_DESIGN_PIDS).read()
        INITIAL_MASSES = calculate_initial_masses(elements, nodes, element_pids)
        non_design_mass = sum(INITIAL_MASSES.get(p, 0.0) for p in NON_DESIGN_PIDS)
        full_design_mass = sum(INITIAL_MASSES.get(p, 0.0) for p in DESIGN_PIDS)
        print(f"     ⚖️ 初始化提取: 满载拓扑域质量 {full_design_mass:.4e} t")
    except Exception as e: 
        print(f"❌ 读取错误: {e}")
        return

    # ---------------------------------------------------------
    # 🟩 [Phase 1] 运行 DiESL 寻优
    # ---------------------------------------------------------
    current_target_domains, _ = extract_matsum_centroids(matsum_path, DESIGN_PIDS)
    print(f"\n{'='*60}\n 🟩 [Phase 1] 启动 DiESL 预测宏观稳定骨架\n{'='*60}")
    
    shutil.copy(k_file_path, run_k_file)
    final_x_phys = run_diesl_double_loop(current_target_domains, work_dir, run_k_file, nodes, elements, element_ids, element_pids, "Phase1_Base_", global_time_logs)
    print("\n   ✅ DiESL 宏观骨架寻优结束。")

    # =========================================================
    # 🎯 【完全满足你的需求】：在 DiESL 结束后立刻生成可视化 K 文件
    # =========================================================
    print("\n   👁️ 正在生成 DiESL 最终结果可视化文件 (动态分配 Part ID)...")
    visual_void_ids = set()
    for pid in DESIGN_PIDS:
        pid_indices = np.where(element_pids == pid)[0]
        n_keep = int(len(pid_indices) * BRIDGE_VOL_RATIO) # 使用设定的体积约束
        visual_void_ids.update(element_ids[pid_indices[idx]] for idx in np.argsort(final_x_phys[pid_indices])[::-1][n_keep:])

    visual_k_file = os.path.join(work_dir, "DiESL_Final_Visual_Separated.k")
    
    with open(k_file_path, 'r') as fin, open(visual_k_file, 'w') as fout:
        in_shell = False
        for line in fin:
            stripped = line.strip()
            if stripped.startswith("*ELEMENT_SHELL"):
                in_shell = True
                fout.write(line)
                continue
            elif stripped.startswith("*"):
                in_shell = False
                fout.write(line)
                continue

            if in_shell and not stripped.startswith("$") and stripped and len(line) >= 16:
                try:
                    eid, current_pid = int(line[0:8]), int(line[8:16])
                    if current_pid in DESIGN_PIDS:
                        # ---------------------------------------------------
                        # 动态分配！
                        # 高密度 (约束以内): current_pid * 100 + 10 (例：1 -> 110, 2 -> 210)
                        # 低密度 (约束以外): current_pid * 100 + 2  (例：1 -> 102, 2 -> 202)
                        # ---------------------------------------------------
                        new_pid = (current_pid * 100 + 2) if eid in visual_void_ids else (current_pid * 100 + 10)
                        fout.write(line[:8] + f"{new_pid:>8}" + line[16:])
                    else:
                        fout.write(line)
                except ValueError:
                    fout.write(line)
            else:
                fout.write(line)
                
    print(f"   ✅ 可视化 K 文件已独立生成: {visual_k_file}")
    print(f"   - 拓扑域单元已根据原 PID 动态拆分 (例: 1->110/102, 2->210/202)")
    # =========================================================


    # ---------------------------------------------------------
    # 🔗 [Bridge] 执行 60% 阈值截断并生成 BESO 初始构型 (硬杀伤)
    # ---------------------------------------------------------
    print(f"\n{'='*60}\n 🔗 [Bridge] 执行 {BRIDGE_VOL_RATIO:.0%} 密度阈值截断 (将虚单元物理剔除)\n{'='*60}")
    
    void_ids_init = set()
    for pid in DESIGN_PIDS:
        pid_indices = np.where(element_pids == pid)[0]
        n_keep = int(len(pid_indices) * BRIDGE_VOL_RATIO)
        void_ids_init.update(element_ids[pid_indices[idx]] for idx in np.argsort(final_x_phys[pid_indices])[::-1][n_keep:])
          
    bridge_k_file = os.path.join(work_dir, "11_BESO_Bridge.k")
    apply_beso_hardkill(k_file_path, bridge_k_file, void_ids_init, DESIGN_PIDS)
    print(f"   ✅ 已生成纯净的两阶段过渡文件：{bridge_k_file}")

    # ---------------------------------------------------------
    # 🟦 [Phase 2] 启动 BESO 真实非线性精修 (双向演化)
    # ---------------------------------------------------------
    print(f"\n{'='*60}\n 🟦 [Phase 2] 启动 BESO 极限吸能雕刻 (真·硬杀伤纯净计算)\n{'='*60}")
    
    centers = np.array([np.mean([nodes[n] for n in elem], axis=0) for elem in elements])
    dists_all = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
    r_min = (np.mean(dists_all) if len(dists_all) > 0 else 3.0) * FILTER_RADIUS_RATIO
  
    vols = np.ones(len(elements)) * MANUAL_THICKNESS
    for i, elem in enumerate(elements):
        try:
            p1, p2 = np.array(nodes[elem[0]]), np.array(nodes[elem[1]])
            p3, p4 = np.array(nodes[elem[2]]), np.array(nodes[elem[3]])
            vols[i] = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p2)) * MANUAL_THICKNESS
        except: pass

    print("   🔍 正在为 BESO 提取全局结构的几何对称性映射关系...")
    beso_sym_groups = build_symmetry_groups(centers, dists_all)

    current_k = bridge_k_file
    history_sens = None
    current_beso_targets = {pid: BRIDGE_VOL_RATIO for pid in DESIGN_PIDS}
    current_er, last_action, vol_history = BESO_ER, None, []
    
    best_valid_iter = -1
    best_valid_SEA = 0.0
    best_valid_k_file = ""

    metrics_csv_path = os.path.join(work_dir, "BESO_Optimization_Metrics.csv")
    csv_headers = ["Iteration", "Solid_Count", "Void_Count", "Volume_Fraction(%)", "Max_Disp_mm", "EA_J", "Mass_t", "SEA_J_t", "DYNA_Time_s", "BESO_Step_Time_s"] + [f"Stage_{i+1}_Force_N" for i in range(len(STAGE_CONSTRAINTS))]
        
    try:
        with open(metrics_csv_path, 'w', encoding='utf-8') as f: f.write(",".join(csv_headers) + "\n")
        print(f"\n   📈 [数据追踪已启动] 将实时记录监控数据至: {metrics_csv_path}")
    except Exception as e: print(f"\n   ❌ 无法创建 CSV 文件，请检查权限: {e}")

    for beso_iter in range(BESO_MAX_ITER):
        t_beso_step_start = time.perf_counter() 

        iter_num = beso_iter + 1
        print(f"\n   [BESO 精修代数 {iter_num}/{BESO_MAX_ITER}] " + "-"*40)
        
        iter_dir = os.path.join(work_dir, f"BESO_Iter_{iter_num:02d}")
        os.makedirs(iter_dir, exist_ok=True)
        iter_run_k = os.path.join(iter_dir, "run_optim.k")
        shutil.copy(current_k, iter_run_k)
        
        print(f"       ⏳ 正在运行纯净物理实体的动力学碰撞验证...")
        
        success, dyna_time = run_dyna_safe(iter_run_k, iter_dir, f"dyna_beso.log")
        print(f"      ⏱️ [耗时统计] LS-DYNA 非线性求解耗时: {dyna_time:.2f} 秒")
        
        stage_forces, D_max, EA, d_curve, f_curve = extract_fd_and_ea(iter_dir, IMPACTOR_NODE_ID, IMPACT_DIRECTION, CONTACT_ID, CONTACT_TYPE, CONTACT_AXIS)
        save_fd_curve_csv(d_curve, f_curve, os.path.join(iter_dir, f"FD_Curve_Iter_{iter_num:02d}.csv"))
        
        d3_path = os.path.join(iter_dir, "d3plot")
        if not os.path.exists(d3_path): 
            print("   ❌ DYNA 在 BESO 阶段未能生成 d3plot，可能硬杀伤引发了计算错误！")
            break
            
        action, force_violation_msg, disp_violation_msg = 'add', "", ""
        
        for idx, (limit_d, limit_f) in enumerate(STAGE_CONSTRAINTS):
             current_f = stage_forces[idx] if idx < len(stage_forces) else 0.0
             if current_f > limit_f:
                 action, force_violation_msg = 'remove', f"阶段 {idx+1} 峰值力 {current_f:.1f}N 超出红线 {limit_f}N"
                 break
               
        if D_max > GLOBAL_DISP_LIMIT: action, disp_violation_msg = 'add', f"最大位移 {D_max:.2f}mm 严重超出全局限制 {GLOBAL_DISP_LIMIT}mm！"

        if not force_violation_msg and not disp_violation_msg:
             if current_er == BESO_ER:
                current_er /= 2.0
                print(f"\n      🎯 【精修模式激活】力与位移约束均已满足！进化率 ER 自动减半至 {current_er:.2%}，开始精细化雕刻...")

        if action == 'remove': print(f"      ⚠️ 警告: {force_violation_msg}。结构偏硬，下一代触发【削减】材料...")
        elif disp_violation_msg: print(f"      🚨 危险: {disp_violation_msg} 结构过软，下一代触发【复活】材料...")
        else: print(f"      🛡️ 物理边界安全。下一代触发【复活】材料以进一步探寻吸能极限...")

        if last_action is not None and action != last_action and current_er > 0.001:
            current_er = max(0.001, current_er / 2.0)
            print(f"      ⚠️ 侦测到边界跨越震荡！触发步长衰减，当前 ER 精确至: {current_er:.3%}")
        
        last_action = action

        for pid in DESIGN_PIDS:
             current_beso_targets[pid] = max(0.10, min(1.0, current_beso_targets[pid] + (current_er if action == 'add' else -current_er)))

        void_list, history_sens = extract_beso_voids(d3_path, current_beso_targets, element_ids, element_pids, centers, vols, r_min, history_sens, IMPACTOR_NODE_ID, IMPACT_DIRECTION, ENABLE_REBOUND_TRUNCATION, sym_groups=beso_sym_groups)
        
        next_k = os.path.join(work_dir, f"run_beso_{iter_num:02d}.k")
        s_count, v_count = apply_beso_hardkill(k_file_path, next_k, void_list, DESIGN_PIDS)
       
        actual_vol = s_count / (s_count + v_count) if (s_count + v_count) > 0 else 0
        vol_history.append(actual_vol)
        
        current_mass = non_design_mass + full_design_mass * actual_vol
        SEA = EA / current_mass if current_mass > 0 else 0.0

        is_valid = not force_violation_msg and not disp_violation_msg
        if is_valid and SEA > best_valid_SEA:
             best_valid_SEA = SEA
             best_valid_iter = iter_num
             best_valid_k_file = iter_run_k
             print(f"      🌟 [新纪录] 发现当前最优可行解！比吸能(SEA)刷新为: {SEA:.2f} J/t")

        print(f"      📊 【网格状态】实单元: {s_count} 个 | 物理剔除: {v_count} 个 | 宏观保留率: {actual_vol:.2%}")
        print(f"      📈 【物理响应】峰值力: [{', '.join([f'S{i+1}={f:.1f}N' for i, f in enumerate(stage_forces)])}] | 最大压缩: {D_max:.2f} mm")
        print(f"      ⚡ 【吸能指标】当前质量: {current_mass:.4e} t | 总吸能(EA): {EA:.2f} J | 比吸能(SEA): {SEA:.2f} J/t")
        
        t_beso_step_end = time.perf_counter() 
        beso_step_time = t_beso_step_end - t_beso_step_start
        
        global_time_logs.append({
            "Phase": "BESO",
            "Macro_Iter": "-",
            "DYNA_Time_s": round(dyna_time, 3),
            "SIMP_Local_Iter": "-",
            "SIMP_Time_s": "-",
            "BESO_Iter": iter_num,
            "BESO_Time_s": round(beso_step_time, 3)
        })

        try:
            with open(metrics_csv_path, 'a', encoding='utf-8') as f:
                row_data = [str(iter_num), str(s_count), str(v_count), f"{actual_vol*100:.2f}", f"{D_max:.4f}", f"{EA:.4f}", f"{current_mass:.6e}", f"{SEA:.4f}", f"{dyna_time:.2f}", f"{beso_step_time:.2f}"] + [f"{f_val:.4f}" for f_val in stage_forces]
                f.write(",".join(row_data) + "\n")
        except Exception as e: print(f"      ❌ CSV 追加写入失败: {e}")
        
        N_window = 5
        if len(vol_history) >= 2 * N_window:
            sum_recent, sum_previous = sum(vol_history[-N_window:]), sum(vol_history[-(2*N_window):-N_window])
            error = abs(sum_recent - sum_previous) / (sum_recent + 1e-9)
            
            if error <= BESO_TAU and current_er < BESO_ER and is_valid:
                print(f"\n      🎉 触发论文级 BESO 绝对收敛！连续 {2*N_window} 代移动平均误差为 {error:.5f} (≤ {BESO_TAU})")
                print(f"      ✅ 最终优化体积完美锁定在: {actual_vol:.2%}，且完全满足所有物理边界约束！")
                break
              
        current_k = next_k

    t_global_end = time.perf_counter()
    total_time = t_global_end - t_global_start
    h, rem = divmod(total_time, 3600)
    m, s = divmod(rem, 60)
    
    excel_path = os.path.join(work_dir, "Optimization_Time_Logs.xlsx")
    csv_path = os.path.join(work_dir, "Optimization_Time_Logs.csv")
    print("\n" + "📊"*20)
    try:
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(global_time_logs)
            df.to_excel(excel_path, index=False)
            print(f" 📈 详细阶段耗时总表已成功导出至 Excel: {excel_path}")
        else:
            raise ImportError
    except:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if global_time_logs:
                writer = csv.DictWriter(f, fieldnames=global_time_logs[0].keys())
                writer.writeheader()
                writer.writerows(global_time_logs)
        print(f" 📈 详细阶段耗时总表已导出为 CSV (请使用 Excel 直接打开): {csv_path}")

    print(f" 📝 完整的控制台 (VS Code) 打印日志已保存至: {log_file_path}")
    print("📊"*20)

    print("\n" + "🏁"*20)
    print(" 🎉 恭喜！两阶段串联优化大功告成！")
    print(f" ⏱️ 任务总耗时: {int(h)}小时 {int(m)}分钟 {s:.2f}秒")
    
    if best_valid_iter != -1:
        final_best_path = os.path.join(work_dir, "FINAL_BEST_BESO.k")
        shutil.copy(best_valid_k_file, final_best_path)
        print(f" 🏆 检索到全局最优模型诞生于 BESO 第 {best_valid_iter} 代！")
        print(f" 🌟 该模型满足所有物理约束，其最优有效比吸能(SEA)为: {best_valid_SEA:.2f} J/t")
        print(f" 💾 最优有效模型 K 文件已独立保存至: {final_best_path}")
    else:
        print(f" ⚠️ 警告：整个 BESO 阶段未能找到严格满足所有约束条件的有效解（模型可能始终过软或过硬），请检查初始设定或放宽约束。")

    print(f" 📊 完整数据记录已安全保存在: {metrics_csv_path}")
    print("🏁"*20 + "\n")

    if hasattr(sys.stdout, 'terminal'):
        sys.stdout = sys.stdout.terminal
  
if __name__ == "__main__":
    main()