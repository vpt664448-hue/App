import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ទំព័រ
st.set_page_config(page_title="Economic Dispatch MATLAB Style", layout="wide")

st.title("⚡ Economic Dispatch (Full Debug Mode)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | NTTI")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Input) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ (Input)")
    pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    # Cost Coefficients
    data = {
        "a": [350.0, 500.0, 600.0],
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Pmin": [100.0, 200.0, 100.0],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_input = st.data_editor(pd.DataFrame(data))

# Loss Coefficients (B-Matrix)
b_mat = [[0.0595, 0.0006, -0.0007], [0.0006, 0.0055, 0.0024], [-0.0007, 0.0024, 0.0088]]
B = np.array(b_mat) / sbase
B0 = np.array([-0.0022, 0.0000, 0.0001])
B00 = 0.000044 * sbase

cost = df_input[["a", "b", "c"]].values
limits = df_input[["Pmin", "Pmax"]].values

if st.button("🚀 គណនា និងបង្ហាញលទ្ធផលលម្អិត", type="primary"):
    
    # --- Iteration 0 ---
    st.markdown("### === Iteration 0 ===")
    num0 = pdt + sum(cost[:,1] / (2 * cost[:,2]))
    den0 = sum(1 / (2 * cost[:,2]))
    lam = num0 / den0
    P = (lam - cost[:,1]) / (2 * cost[:,2])
    
    st.write(f"**Lambda initial** = `{lam:.9f}`")
    st.write(f"**P1** = {P[0]:.9f} MW, **P2** = {P[1]:.9f} MW, **P3** = {P[2]:.9f} MW")
    st.divider()

    # --- Iteration 1 ---
    st.markdown("### === Iteration 1 ===")
    P_prev = P.copy()
    # Gauss-Seidel Update
    P[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_prev[1] + B[0,2]*P_prev[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P[0] + B[1,2]*P_prev[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P[0] + B[2,1]*P[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    # Calculate Loss and DP
    PL = P @ B @ P.T + B0 @ P.T + B00
    DP = pdt + PL - sum(P)

    # Calculate Gradients (F, X, Y)
    F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P[1]+B[0,2]*P[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P[0]+B[1,2]*P[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P[0]+B[2,1]*P[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam = DP / (F + X + Y)
    
    # បង្ហាញ F, X, Y ជា Column
    col1, col2, col3 = st.columns(3)
    col1.metric("Value of F", f"{F:.9f}")
    col2.metric("Value of X", f"{X:.9f}")
    col3.metric("Value of Y", f"{Y:.9f}")
    
    st.success(f"✅ **dlam1** = `{dlam:.9f}`")
    
    lam = lam + dlam
    st.info(f"**New Lambda** = `{lam:.9f}`")
    st.write(f"**P1** = {P[0]:.9f}, **P2** = {P[1]:.9f}, **P3** = {P[2]:.9f}")
    st.divider()

    # --- Iteration 2 ---
    st.markdown("### === Iteration 2 ===")
    P_prev = P.copy()
    P[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_prev[1] + B[0,2]*P_prev[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P[0] + B[1,2]*P_prev[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P[0] + B[2,1]*P[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    # Check Limit for Unit 3
    hit3 = 0
    if P[2] > limits[2, 1]:
        P[2] = limits[2, 1]
        hit3 = 1
        st.warning(f"⚠️ P3 លើសកម្រិតកំណត់ (Hit Pmax: {limits[2, 1]} MW)")

    PL = P @ B @ P.T + B0 @ P.T + B00
    DP = pdt + PL - sum(P)

    F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P[1]+B[0,2]*P[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P[0]+B[1,2]*P[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    # បើ Unit 3 ជាប់ Limit នោះ Gradient Y = 0
    Y = 0 if hit3 else (cost[2,2]*(1-B0[2]-2*(B[2,0]*P[0]+B[2,1]*P[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam = DP / (F + X + Y)
    
    col1_2, col2_2, col3_2 = st.columns(3)
    col1_2.metric("Value of F", f"{F:.9f}")
    col2_2.metric("Value of X", f"{X:.9f}")
    col3_2.metric("Value of Y", f"{Y:.9f}")
    
    st.success(f"✅ **dlam2** = `{dlam:.9f}`")
    
    lam = lam + dlam
    st.info(f"**New Lambda** = `{lam:.9f}`")
    st.write(f"**P1** = {P[0]:.9f}, **P2** = {P[1]:.9f}, **P3** = {P[2]:.9f}")
    st.divider()

    # --- លទ្ធផលចុងក្រោយ ---
    st.subheader("🏆 លទ្ធផលចុងក្រោយ (Final Result)")
    total_p = sum(P)
    gencost = sum(cost[:,0] + cost[:,1]*P + cost[:,2]*(P**2))
    
    res_col1, res_col2 = st.columns(2)
    res_col1.metric("Total Generation (MW)", f"{total_p:.4f}")
    res_col2.metric("Total Cost ($/h)", f"{gencost:.4f}")
    
    st.balloons()
