import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ទូទៅរបស់ App
st.set_page_config(page_title="Economic Dispatch IT3 - BROSNI 168", layout="wide")
st.title("⚡ Economic Dispatch with B-Matrix (Iteration 3)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | និស្សិតវិស្វកម្មអគ្គិសនី NTTI")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Input Section) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ Input")
    pdt_input = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase_input = st.number_input("Sbase (MVA):", value=100.0)
    
    st.subheader("1. Cost Coefficients & Limits")
    # តារាងបញ្ចូល a, b, c និង MW Limits
    default_costs = {
        "a ($/h)": [350.0, 500.0, 600.0],
        "b ($/MWh)": [7.2, 7.3, 7.8],
        "c ($/MW^2h)": [0.004, 0.0025, 0.003],
        "Pmin": [100.0, 200.0, 100.0],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_cost = st.data_editor(pd.DataFrame(default_costs), num_rows="fixed")

# ផ្នែកបញ្ចូល Matrix នៅកណ្ដាល
st.subheader("🔧 Loss Coefficients (B-Matrix, B0i, B00)")
col_m1, col_m2 = st.columns([2, 1])

with col_m1:
    st.write("**B Matrix (pu):**")
    b_pu_default = [[0.0595, 0.0006, -0.0007],
                    [0.0006, 0.0055, 0.0024],
                    [-0.0007, 0.0024, 0.0088]]
    df_b_matrix = st.data_editor(pd.DataFrame(b_pu_default, columns=['G1', 'G2', 'G3'], index=['G1', 'G2', 'G3']))

with col_m2:
    st.write("**B0i & B00 (pu):**")
    b01 = st.number_input("B01:", value=-0.0022, format="%.6f")
    b02 = st.number_input("B02:", value=0.0000, format="%.6f")
    b03 = st.number_input("B03:", value=0.0001, format="%.6f")
    b00_val = st.number_input("B00:", value=0.000044, format="%.6f")

# រៀបចំទិន្នន័យសម្រាប់គណនា
cost = df_cost[["a ($/h)", "b ($/MWh)", "c ($/MW^2h)"]].values
limits = df_cost[["Pmin", "Pmax"]].values
B = df_b_matrix.values / sbase_input
B0 = np.array([b01, b02, b03])
B00 = b00_val * sbase_input

# --- ដំណើរការគណនាពេលចុចប៊ូតុង ---
if st.button("🚀 ចាប់ផ្ដើមគណនាដល់ IT3", type="primary"):
    
    # Iteration 0
    st.divider()
    st.subheader("📍 Iteration 0")
    num0 = pdt_input + sum(cost[:, 1] / (2 * cost[:, 2]))
    den0 = sum(1 / (2 * cost[:, 2]))
    lam = num0 / den0
    P0 = (lam - cost[:, 1]) / (2 * cost[:, 2])
    
    res0_df = pd.DataFrame({"Generator": ["G1", "G2", "G3"], "Power (MW)": P0})
    st.table(res0_df)
    st.write(f"**Lambda (λ) Initial:** {lam:.9f}")

    # បង្កើត Loop សម្រាប់ IT1 ដល់ IT3
    P_prev = P0.copy()
    hit_status = [False, False, False]

    for i in range(1, 4):
        st.subheader(f"📍 Iteration {i}")
        P_curr = np.zeros(3)
        
        # គណនា P1, P2, P3 តាម Gauss-Seidel
        for j in range(3):
            if hit_status[j]:
                P_curr[j] = limits[j, 1] # ប្រសិនបើជាប់ Limit
            else:
                sum_loss = 0
                for k in range(3):
                    if k != j:
                        # ប្រើតម្លៃថ្មីបើមាន (Gauss-Seidel)
                        p_val = P_curr[k] if k < j else P_prev[k]
                        sum_loss += B[j, k] * p_val
                
                # រូបមន្ត Economic Dispatch ជាមួយ Loss
                P_curr[j] = (lam * (1 - B0[j] - 2 * sum_loss) - cost[j, 1]) / (2 * (cost[j, 2] + lam * B[j, j]))

        # ឆែកលក្ខខណ្ឌ Limit ដូចក្នុង Matlab
        if i == 2 and P_curr[2] > limits[2, 1]:
            P_curr[2] = limits[2, 1]
            hit_status[2] = True
        if i == 3 and P_curr[1] > limits[1, 1]:
            P_curr[1] = limits[1, 1]
            hit_status[1] = True

        # គណនា PL, DP, និងមេគុណ Gradient (F, X, Y)
        PL = P_curr @ B @ P_curr.T + B0 @ P_curr.T + B00
        DP = pdt_input + PL - sum(P_curr)
        
        gradients = []
        for j in range(3):
            if hit_status[j]:
                gradients.append(0)
            else:
                sum_g = 0
                for k in range(3):
                    if k != j: sum_g += B[j, k] * P_curr[k]
                
                grad = (cost[j, 2]*(1 - B0[j] - 2*sum_g) + B[j, j]*cost[j, 1]) / (2*(cost[j, 2] + lam * B[j, j])**2)
                gradients.append(grad)
        
        Dlam = DP / sum(gradients)
        lam_new = lam + Dlam
        
        # បង្ហាញលទ្ធផលក្នុង IT នីមួយៗ
        c1, c2 = st.columns(2)
        with c1:
            st.code(f"P1 = {P_curr[0]:.9f} MW\nP2 = {P_curr[1]:.9f} MW\nP3 = {P_curr[2]:.9f} MW\nPL = {PL:.9f} MW")
        with c2:
            st.code(f"DP = {DP:.9f} MW\nDlam = {Dlam:.9f}\nLambda_new = {lam_new:.9f}")
        
        # រៀបចំសម្រាប់ជំហានបន្ទាប់
        lam = lam_new
        P_prev = P_curr.copy()

    # លទ្ធផលចុងក្រោយ
    st.divider()
    final_cost = sum(cost[:, 0] + cost[:, 1] * P_curr + cost[:, 2] * (P_curr**2))
    st.success(f"💰 **តម្លៃចំណាយសរុបចុងក្រោយ (Total Generation Cost): {final_cost:,.4f} $/h**")
