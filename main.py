import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ App
st.set_page_config(page_title="Economic Dispatch MATLAB Style", layout="wide")
st.title("⚡ Economic Dispatch (Gauss-Seidel MATLAB Logic)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | NTTI")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Input) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ (Input)")
    pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    st.subheader("1. Cost Coefficients & Limits")
    # ទិន្នន័យដើមតាម MATLAB
    data = {
        "a": [350.0, 500.0, 600.0],
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Pmin": [100.0, 200.0, 100.0],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_input = st.data_editor(pd.DataFrame(data))

st.subheader("🔧 Loss Coefficients (B-Matrix)")
col_b1, col_b2 = st.columns([2, 1])

with col_b1:
    b_mat = [[0.0595, 0.0006, -0.0007], [0.0006, 0.0055, 0.0024], [-0.0007, 0.0024, 0.0088]]
    df_b = st.data_editor(pd.DataFrame(b_mat, columns=['G1','G2','G3'], index=['G1','G2','G3']))

with col_b2:
    b01 = st.number_input("B01:", value=-0.0022, format="%.6f")
    b02 = st.number_input("B02:", value=0.0000, format="%.6f")
    b03 = st.number_input("B03:", value=0.0001, format="%.6f")
    b00_in = st.number_input("B00:", value=0.000044, format="%.6f")

# រៀបចំ Arrays សម្រាប់គណនា
cost = df_input[["a", "b", "c"]].values
limits = df_input[["Pmin", "Pmax"]].values
B = df_b.values / sbase
B0 = np.array([b01, b02, b03])
B00 = b00_in * sbase

# --- ចុចប៊ូតុងគណនា ---
if st.button("🚀 គណនា (Run MATLAB Logic)", type="primary"):
    
    # --- Iteration 0 ---
    st.markdown("### === Iteration 0 ===")
    num0 = pdt + sum(cost[:,1] / (2 * cost[:,2]))
    den0 = sum(1 / (2 * cost[:,2]))
    lam = num0 / den0
    P0 = (lam - cost[:,1]) / (2 * cost[:,2])
    
    st.write(f"**Lambda** = {lam:.9f}")
    st.write(f"**P1** = {P0[0]:.9f} MW, **P2** = {P0[1]:.9f} MW, **P3** = {P0[2]:.9f} MW")
    st.divider()

    # --- Iteration 1 ---
    st.markdown("### === Iteration 1 ===")
    P1 = np.zeros(3)
    P1[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P0[1] + B[0,2]*P0[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P1[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P1[0] + B[1,2]*P0[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P1[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P1[0] + B[2,1]*P1[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    PL1 = P1 @ B @ P1.T + B0 @ P1.T + B00
    DP1 = pdt + PL1 - sum(P1)

    F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P1[1]+B[0,2]*P1[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P1[0]+B[1,2]*P1[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P1[0]+B[2,1]*P1[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam1 = DP1 / (F + X + Y)
    lam = lam + dlam1

    st.write(f"**P1** = {P1[0]:.9f} MW, **P2** = {P1[1]:.9f} MW, **P3** = {P1[2]:.9f} MW")
    st.write(f"**PL** = {PL1:.9f} MW, **DP** = {DP1:.9f} MW")
    st.write(f"**F** = {F:.9f}, **X** = {X:.9f}, **Y** = {Y:.9f}")
    st.write(f"**Lambda** = {lam:.9f}")
    st.divider()

    # --- Iteration 2 ---
    st.markdown("### === Iteration 2 ===")
    P2 = np.zeros(3)
    P2[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P1[1] + B[0,2]*P1[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P2[1] = (lam*(1 - B0[1] - 2*(B[2,0]*P2[0] + B[1,2]*P1[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P2[2] = (lam*(1 - B0[2] - 2*(B[3,0]*P2[0] + B[2,1]*P2[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    st.write(f"**P1** = {P2[0]:.9f}, **P2** = {P2[1]:.9f}, **P3** = {P2[2]:.9f}")

    hit3 = 0
    if P2[2] > limits[2, 1]:
        P2[2] = limits[2, 1]
        hit3 = 1
        st.warning(f"⚠️ P3_new = {P2[2]:.9f} MW (Limit Hit)")

    PL2 = P2 @ B @ P2.T + B0 @ P2.T + B00
    DP2 = pdt + PL2 - sum(P2)

    F2 = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P2[1]+B[0,2]*P2[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X2 = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P2[0]+B[1,2]*P2[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y2 = 0 if hit3 else (cost[2,2]*(1-B0[2]-2*(B[2,0]*P2[0]+B[2,1]*P2[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam2 = DP2 / (F2 + X2 + Y2)
    lam = lam + dlam2

    st.write(f"**PL** = {PL2:.9f}, **DP** = {DP2:.9f}, **Lambda** = {lam:.9f}")
    st.divider()

    # --- Iteration 3 ---
    st.markdown("### === Iteration 3 ===")
    P3 = np.zeros(3)
    P3[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P2[1] + B[0,2]*P2[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P3[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P3[0] + B[1,2]*P2[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    
    st.write(f"**P1** = {P3[0]:.9f}, **P2** = {P3[1]:.9f}")

    hit2 = 0
    if P3[1] > limits[1, 1]:
        P3[1] = limits[1, 1]
        hit2 = 1
        st.warning(f"⚠️ P2_new = {P3[1]:.9f} MW (Limit Hit)")
    
    P3[2] = limits[2, 1] # Fixed
    st.info(f"ℹ️ P3 = {P3[2]:.9f} MW (Fixed)")

    PL3 = P3 @ B @ P3.T + B0 @ P3.T + B00
    DP3 = pdt + PL3 - sum(P3)

    F3 = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P3[1]+B[0,2]*P3[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X3 = 0 if hit2 else (cost[1,2]*(1-B0[1]-2*(B[1,0]*P3[0]+B[1,2]*P3[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y3 = 0

    dlam3 = DP3 / (F3 + X3 + Y3)
    lam = lam + dlam3

    st.write(f"**PL** = {PL3:.9f}, **DP** = {DP3:.9f}, **Lambda** = {lam:.9f}")

    # --- Final Result ---
    st.divider()
    st.subheader("🏆 FINAL RESULT")
    gencost = sum(cost[:,0] + cost[:,1]*P3 + cost[:,2]*(P3**2))
    st.success(f"**Gencost = {gencost:.9f} $/h**")
