import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ទូទៅរបស់ App
st.set_page_config(page_title="Economic Dispatch - BROSNI 168", layout="wide")
st.title("⚡ Economic Dispatch Calculation (IT3)")
st.write(f"អ្នករៀបចំ៖ **{st.session_state.get('user_name', 'ផាត ប្រុសនិ')}** (BROSNI 168)")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Sidebar) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ")
    pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    st.subheader("Cost & Limits")
    default_costs = {
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_cost = st.data_editor(pd.DataFrame(default_costs))

# --- ផ្នែកបញ្ចូល Matrix (B, B0i, B00) ---
st.subheader("🔧 Loss Coefficients")
c_m1, c_m2 = st.columns([2, 1])

with c_m1:
    st.write("**B Matrix (pu):**")
    b_vals = [[0.0595, 0.0006, -0.0007], [0.0006, 0.0055, 0.0024], [-0.0007, 0.0024, 0.0088]]
    df_b = st.data_editor(pd.DataFrame(b_vals, columns=['G1','G2','G3'], index=['G1','G2','G3']))

with c_m2:
    st.write("**B0i & B00 (pu):**")
    b01 = st.number_input("B01:", value=-0.0022, format="%.6f")
    b02 = st.number_input("B02:", value=0.0000, format="%.6f")
    b03 = st.number_input("B03:", value=0.0001, format="%.6f")
    b00 = st.number_input("B00:", value=0.000044, format="%.6f")

# រៀបចំទិន្នន័យ
cost_b = df_cost["b"].values
cost_c = df_cost["c"].values
pmax = df_cost["Pmax"].values
B = df_b.values / sbase
B0 = np.array([b01, b02, b03])
B00_val = b00 * sbase

# --- គណនាពេលចុចប៊ូតុង ---
if st.button("🚀 គណនាឥឡូវនេះ", type="primary"):
    
    # Iteration 0
    lam = (pdt + sum(cost_b / (2 * cost_c))) / sum(1 / (2 * cost_c))
    P_prev = (lam - cost_b) / (2 * cost_c)
    
    # Loop IT1 ដល់ IT3
    hit_status = [False, False, False]
    
    for i in range(1, 4):
        st.markdown(f"### 📍 Iteration {i}")
        P_curr = np.zeros(3)
        
        # Gauss-Seidel
        for j in range(3):
            if hit_status[j]: P_curr[j] = pmax[j]
            else:
                sum_L = sum(B[j, k] * (P_curr[k] if k < j else P_prev[k]) for k in range(3) if k != j)
                P_curr[j] = (lam*(1-B0[j]-2*sum_L) - cost_b[j]) / (2*(cost_c[j] + lam*B[j,j]))

        # Limit Check (ដូចក្នុង MATLAB)
        if i == 2 and P_curr[2] > pmax[2]: 
            P_curr[2] = pmax[2]; hit_status[2] = True
        if i == 3 and P_curr[1] > pmax[1]: 
            P_curr[1] = pmax[1]; hit_status[1] = True

        PL = P_curr @ B @ P_curr.T + B0 @ P_curr.T + B00_val
        DP = pdt + PL - sum(P_curr)
        
        # គណនា Gradient F, X, Y
        grads = []
        for j in range(3):
            if hit_status[j]: grads.append(0)
            else:
                sum_g = sum(B[j, k] * P_curr[k] for k in range(3) if k != j)
                grads.append((cost_c[j]*(1-B0[j]-2*sum_g) + B[j,j]*cost_b[j]) / (2*(cost_c[j] + lam*B[j,j])**2))
        
        dlam = DP / sum(grads)
        
        # --- បង្ហាញលទ្ធផលឱ្យស្អាត (មិនឱ្យបង្រួមលេខ) ---
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.info("**លទ្ធផលកម្លាំងអគ្គិសនី (Power Output):**")
            res_table = pd.DataFrame({
                "Parameter": ["P1 (MW)", "P2 (MW)", "P3 (MW)", "PL (Loss)"],
                "Value": [f"{P_curr[0]:.6f}", f"{P_curr[1]:.6f}", f"{P_curr[2]:.6f}", f"{PL:.6f}"]
            })
            st.table(res_table)
            
        with col_res2:
            st.warning("**លទ្ធផល Lambda & Error:**")
            err_table = pd.DataFrame({
                "Parameter": ["DP (Error)", "Delta Lambda", "New Lambda"],
                "Value": [f"{DP:.6f}", f"{dlam:.6f}", f"{(lam + dlam):.6f}"]
            })
            st.table(err_table)
            
        lam += dlam
        P_prev = P_curr.copy()
        st.divider()

    st.success(f"✅ **ការគណនាបានសម្រេច! ប្អូនអាចយកលទ្ធផលនេះទៅប្រើប្រាស់បាន។**")
