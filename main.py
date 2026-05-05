import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ App
st.set_config(page_title="Economic Dispatch MATLAB Style", layout="wide")
st.title("⚡ Economic Dispatch (Full Debug Mode)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | NTTI")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Input) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ (Input)")
    pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    data = {
        "a": [350.0, 500.0, 600.0],
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Pmin": [100.0, 200.0, 100.0],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_input = st.data_editor(pd.DataFrame(data))

# Loss Coefficients
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
    
    st.write(f"**Lambda** = {lam:.9f}")
    st.write(f"**P1** = {P[0]:.9f}, **P2** = {P[1]:.9f}, **P3** = {P[2]:.9f}")
    st.divider()

    # --- Iteration 1 ---
    st.markdown("### === Iteration 1 ===")
    P_prev = P.copy()
    P[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_prev[1] + B[0,2]*P_prev[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P[0] + B[1,2]*P_prev[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P[0] + B[2,1]*P[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    PL = P @ B @ P.T + B0 @ P.T + B00
    DP = pdt + PL - sum(P)

    F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P[1]+B[0,2]*P[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P[0]+B[1,2]*P[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P[0]+B[2,1]*P[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam = DP / (F + X + Y)
    
    st.info(f"**DEBUG:** F={F:.9f}, X={X:.9f}, Y={Y:.9f}")
    st.success(f"**dlam1** = {dlam:.9f}")
    
    lam = lam + dlam
    st.write(f"**New Lambda** = {lam:.9f}")
    st.write(f"**P1** = {P[0]:.9f}, **P2** = {P[1]:.9f}, **P3** = {P[2]:.9f}")
    st.divider()

    # --- Iteration 2 ---
    st.markdown("### === Iteration 2 ===")
    P_prev = P.copy()
    P[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_prev[1] + B[0,2]*P_prev[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
    P[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P[0] + B[1,2]*P_prev[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
    P[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P[0] + B[2,1]*P[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

    hit3 = 0
    if P[2] > limits[2, 1]:
        P[2] = limits[2, 1]
        hit3 = 1
        st.warning(f"⚠️ P3 hit Pmax!")

    PL = P @ B @ P.T + B0 @ P.T + B00
    DP = pdt + PL - sum(P)

    F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P[1]+B[0,2]*P[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2)
    X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P[0]+B[1,2]*P[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2)
    Y = 0 if hit3 else (cost[2,2]*(1-B0[2]-2*(B[2,0]*P[0]+B[2,1]*P[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2)

    dlam = DP / (F + X + Y)
    
    st.info(f"**DEBUG:** F={F:.9f}, X={X:.9f}, Y={Y:.9f}")
    st.success(f"**dlam2** = {dlam:.9f}")
    
    lam = lam + dlam
    st.write(f"**New Lambda** = {lam:.9f}")
    st.write(f"**P1** = {P[0]:.9f}, **P2** = {P[1]:.9f}, **P3** = {P[2]:.9f}")
    st.divider()

    # --- Final Output ---
    st.subheader("🏆 Final Result")
    gencost = sum(cost[:,0] + cost[:,1]*P + cost[:,2]*(P**2))
    st.metric("Total Generation Cost", f"{gencost:.4f} $/h")
