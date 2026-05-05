import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Economic Dispatch - BROSNI 168", layout="wide")
st.title("⚡ Economic Dispatch with B-Matrix (Gauss-Seidel)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | គណនាលម្អិតដល់ Iteration 3 ដូច MATLAB")

# --- ផ្នែកបញ្ចូលទិន្នន័យ ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ")
    Pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    Sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    st.subheader("Cost Coefficients (a, b, c)")
    cost_data = pd.DataFrame({
        "a": [350.0, 500.0, 600.0],
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Min": [100.0, 200.0, 100.0],
        "Max": [200.0, 400.0, 200.0]
    })
    df_cost = st.data_editor(cost_data, use_container_width=True)
    
    st.subheader("Loss Coefficients (B Matrix pu)")
    b_pu_data = pd.DataFrame([
        [0.0595,  0.0006, -0.0007],
        [0.0006,  0.0055,  0.0024],
        [-0.0007, 0.0024,  0.0088]
    ])
    df_b = st.data_editor(b_pu_data, use_container_width=True)

# ទាញយកទិន្នន័យមកប្រើ
cost = df_cost[["a", "b", "c"]].values
mwlimits = df_cost[["Min", "Max"]].values

B_pu = df_b.values
B0_pu = np.array([-0.0022, 0.0000, 0.0001])
B00_pu = 0.000044

B = B_pu / Sbase
B0 = B0_pu
B00 = B00_pu * Sbase

# --- ចាប់ផ្តើមការគណនាពេលចុចប៊ូតុង ---
if st.button("🚀 គណនា (Calculate like MATLAB)", type="primary"):
    
    st.divider()
    
    # ==========================================
    # Iteration 0
    # ==========================================
    st.subheader("=== Iteration 0 ===")
    num = Pdt + sum(cost[:, 1] / (2 * cost[:, 2]))
    den = sum(1 / (2 * cost[:, 2]))
    lam = num / den

    P_0 = (lam - cost[:, 1]) / (2 * cost[:, 2])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lambda", f"{lam:.9f}")
    c2.metric("P1", f"{P_0[0]:.9f} MW")
    c3.metric("P2", f"{P_0[1]:.9f} MW")
    c4.metric("P3", f"{P_0[2]:.9f} MW")

    # ==========================================
    # Iteration 1
    # ==========================================
    st.subheader("=== Iteration 1 ===")
    P_1 = np.zeros(3)
    P_1[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_0[1] + B[0,2]*P_0[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P_1[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P_1[0] + B[1,2]*P_0[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P_1[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P_1[0] + B[2,1]*P_1[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    PL1 = P_1 @ B @ P_1.T + B0 @ P_1.T + B00
    DP1 = Pdt + PL1 - sum(P_1)

    F1 = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P_1[1]+B[0,2]*P_1[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X1 = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P_1[0]+B[1,2]*P_1[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y1 = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P_1[0]+B[2,1]*P_1[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    Dlam1 = DP1 / (F1 + X1 + Y1)
    lam = lam + Dlam1

    col1, col2 = st.columns(2)
    with col1:
        st.code(f"P1 = {P_1[0]:.9f} MW\nP2 = {P_1[1]:.9f} MW\nP3 = {P_1[2]:.9f} MW\nPL = {PL1:.9f} MW\nDP = {DP1:.9f} MW")
    with col2:
        st.code(f"F = {F1:.9f}\nX = {X1:.9f}\nY = {Y1:.9f}\nDlam = {Dlam1:.9f}\nLambda = {lam:.9f}")

    # ==========================================
    # Iteration 2
    # ==========================================
    st.subheader("=== Iteration 2 ===")
    P_2 = np.zeros(3)
    P_2[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_1[1] + B[0,2]*P_1[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P_2[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P_2[0] + B[1,2]*P_1[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P_2[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P_2[0] + B[2,1]*P_2[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    hit3 = 0
    p3_status = f"{P_2[2]:.9f} MW"
    if P_2[2] > mwlimits[2, 1]:
        P_2[2] = mwlimits[2, 1]
        hit3 = 1
        p3_status = f"{P_2[2]:.9f} MW (Limit Hit)"

    PL2 = P_2 @ B @ P_2.T + B0 @ P_2.T + B00
    DP2 = Pdt + PL2 - sum(P_2)

    F2 = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P_2[1]+B[0,2]*P_2[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X2 = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P_2[0]+B[1,2]*P_2[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    if hit3:
        Y2 = 0
    else:
        Y2 = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P_2[0]+B[2,1]*P_2[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    Dlam2 = DP2 / (F2 + X2 + Y2)
    lam = lam + Dlam2

    col1, col2 = st.columns(2)
    with col1:
        st.code(f"P1 = {P_2[0]:.9f} MW\nP2 = {P_2[1]:.9f} MW\nP3 = {p3_status}\nPL = {PL2:.9f} MW\nDP = {DP2:.9f} MW")
    with col2:
        st.code(f"F = {F2:.9f}\nX = {X2:.9f}\nY = {Y2:.9f}\nDlam = {Dlam2:.9f}\nLambda = {lam:.9f}")

    # ==========================================
    # Iteration 3
    # ==========================================
    st.subheader("=== Iteration 3 ===")
    P_3 = np.zeros(3)
    P_3[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_2[1] + B[0,2]*P_2[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P_3[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P_3[0] + B[1,2]*P_2[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))

    hit2 = 0
    p2_status = f"{P_3[1]:.9f} MW"
    if P_3[1] > mwlimits[1, 1]:
        P_3[1] = mwlimits[1, 1]
        hit2 = 1
        p2_status = f"{P_3[1]:.9f} MW (Limit Hit)"
        
    P_3[2] = mwlimits[2, 1]  # P3 នៅជាប់ Limit ពី IT2
    p3_status = f"{P_3[2]:.9f} MW (Limit Hit)"

    PL3 = P_3 @ B @ P_3.T + B0 @ P_3.T + B00
    DP3 = Pdt + PL3 - sum(P_3)

    F3 = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P_3[1]+B[0,2]*P_3[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    if hit2:
        X3 = 0
    else:
        X3 = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P_3[0]+B[1,2]*P_3[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    
    Y3 = 0 # ព្រោះ P3 ជាប់ Limit រួចហើយ

    Dlam3 = DP3 / (F3 + X3 + Y3)
    lam = lam + Dlam3

    col1, col2 = st.columns(2)
    with col1:
        st.code(f"P1 = {P_3[0]:.9f} MW\nP2 = {p2_status}\nP3 = {p3_status}\nPL = {PL3:.9f} MW\nDP = {DP3:.9f} MW")
    with col2:
        st.code(f"F = {F3:.9f}\nX = {X3:.9f}\nY = {Y3:.9f}\nDlam = {Dlam3:.9f}\nLambda = {lam:.9f}")

    # ==========================================
    # Final Result
    # ==========================================
    st.divider()
    st.subheader("=== FINAL RESULT ===")
    Gencost = sum(cost[:, 0] + cost[:, 1]*P_3 + cost[:, 2]*(P_3**2))
    st.success(f"💰 **Total Generation Cost = {Gencost:,.9f} $/h**")
