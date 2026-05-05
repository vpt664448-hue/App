import streamlit as st
import numpy as np
import pandas as pd

# កំណត់ទម្រង់ទំព័រ
st.set_page_config(page_title="Economic Dispatch 3-IT Mode", layout="wide")

st.title("⚡ Economic Dispatch (Limit Check from IT2)")
st.write("អភិវឌ្ឍន៍ដោយ៖ **ផាត ប្រុសនិ (BROSNI 168)** | NTTI")

# --- ផ្នែកបញ្ចូលទិន្នន័យ (Input) ---
with st.sidebar:
    st.header("📥 បញ្ចូលទិន្នន័យ (Input)")
    pdt = st.number_input("Power Demand (Pdt) MW:", value=740.0)
    sbase = st.number_input("Sbase (MVA):", value=100.0)
    
    st.subheader("1. Cost Coefficients & Limits")
    data = {
        "a": [350.0, 500.0, 600.0],
        "b": [7.2, 7.3, 7.8],
        "c": [0.004, 0.0025, 0.003],
        "Pmin": [100.0, 200.0, 100.0],
        "Pmax": [200.0, 400.0, 200.0]
    }
    df_input = st.data_editor(pd.DataFrame(data))

    st.subheader("2. Loss Coefficients B0i & B00")
    b01 = st.number_input("B01:", value=-0.0022, format="%.6f")
    b02 = st.number_input("B02:", value=0.0000, format="%.6f")
    b03 = st.number_input("B03:", value=0.0001, format="%.6f")
    b00_raw = st.number_input("B00:", value=0.000044, format="%.6f")

# Matrix B (Bij)
b_mat = [[0.0595, 0.0006, -0.0007], [0.0006, 0.0055, 0.0024], [-0.0007, 0.0024, 0.0088]]
B = np.array(b_mat) / sbase
B0 = np.array([b01, b02, b03])
B00 = b00_raw * sbase

cost = df_input[["a", "b", "c"]].values
limits = df_input[["Pmin", "Pmax"]].values

if st.button("🚀 គណនា 3 Iterations", type="primary"):
    
    # --- Iteration 0 ---
    st.markdown("### === Iteration 0 ===")
    num0 = pdt + sum(cost[:,1] / (2 * cost[:,2]))
    den0 = sum(1 / (2 * cost[:,2]))
    lam = num0 / den0
    P = (lam - cost[:,1]) / (2 * cost[:,2])
    st.write(f"**Lambda initial** = `{lam:.9f}`")
    st.write(f"**P1** = {P[0]:.6f}, **P2** = {P[1]:.6f}, **P3** = {P[2]:.6f}")
    st.divider()

    for i in range(1, 4):
        st.markdown(f"### === Iteration {i} ===")
        P_prev = P.copy()
        hit = [0, 0, 0]
        
        # 1. គណនា P1, P2, P3 ជាមុនសិន
        P[0] = (lam*(1 - B0[0] - 2*(B[0,1]*P_prev[1] + B[0,2]*P_prev[2])) - cost[0,1]) / (2*(cost[0,2] + lam*B[0,0]))
        P[1] = (lam*(1 - B0[1] - 2*(B[1,0]*P[0] + B[1,2]*P_prev[2])) - cost[1,1]) / (2*(cost[1,2] + lam*B[1,1]))
        P[2] = (lam*(1 - B0[2] - 2*(B[2,0]*P[0] + B[2,1]*P[1])) - cost[2,1]) / (2*(cost[2,2] + lam*B[2,2]))

        # 2. ចាប់ផ្ដើមឆែកលក្ខខណ្ឌ Limit ចាប់ពី IT2 ឡើងទៅ
        if i >= 2:
            for j in range(3):
                if P[j] > limits[j, 1]:
                    P[j] = limits[j, 1]
                    hit[j] = 1
                    st.warning(f"⚠️ P{j+1} ជាប់ Limit Pmax ({limits[j, 1]})")
                elif P[j] < limits[j, 0]:
                    P[j] = limits[j, 0]
                    hit[j] = 1
                    st.warning(f"⚠️ P{j+1} ជាប់ Limit Pmin ({limits[j, 0]})")

        # 3. គណនា Loss និង DP បន្ទាប់ពីបាន P ពេញលេញ
        PL = P @ B @ P.T + B0 @ P.T + B00
        DP = pdt + PL - sum(P)

        # 4. គណនា Gradients (F, X, Y) បន្ទាប់ពីរក P1, P2, P3 រួច
        F = (cost[0,2]*(1-B0[0]-2*(B[0,1]*P[1]+B[0,2]*P[2])) + B[0,0]*cost[0,1]) / (2*(cost[0,2]+lam*B[0,0])**2) if not hit[0] else 0
        X = (cost[1,2]*(1-B0[1]-2*(B[1,0]*P[0]+B[1,2]*P[2])) + B[1,1]*cost[1,1]) / (2*(cost[1,2]+lam*B[1,1])**2) if not hit[1] else 0
        Y = (cost[2,2]*(1-B0[2]-2*(B[2,0]*P[0]+B[2,1]*P[1])) + B[2,2]*cost[2,1]) / (2*(cost[2,2]+lam*B[2,2])**2) if not hit[2] else 0

        dlam = DP / (F + X + Y)
        
        # បង្ហាញតម្លៃ
        st.write(f"**P-Current:** P1={P[0]:.6f}, P2={P[1]:.6f}, P3={P[2]:.6f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Gradient F", f"{F:.9f}")
        c2.metric("Gradient X", f"{X:.9f}")
        c3.metric("Gradient Y", f"{Y:.9f}")
        
        st.success(f"✅ **dlam{i}** = `{dlam:.9f}` | **DP** = `{DP:.6f}`")
        lam = lam + dlam
        st.info(f"**Lambda ថ្មី** = `{lam:.9f}`")
        st.divider()

    # --- Final Output ---
    st.subheader("🏆 លទ្ធផលចុងក្រោយ")
    gencost = sum(cost[:,0] + cost[:,1]*P + cost[:,2]*(P**2))
    st.metric("Total Generation Cost", f"{gencost:.4f} $/h")
