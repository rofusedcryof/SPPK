import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import math

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(page_title="SPPK Matchmaker Pro", page_icon="💘", layout="wide")

# Inisialisasi State untuk Menyimpan Data Kandidat
if 'candidates_data' not in st.session_state:
    st.session_state['candidates_data'] = []

# ==============================================================================
# DATA REFERENSI LOKASI (JOGJA & SEKITARNYA)
# ==============================================================================
loc_options = {
    "Jogja (Kota)": (-7.7956, 110.3695),
    "Sleman": (-7.7145, 110.3456),
    "Bantul": (-7.8924, 110.3323),
    "Gunungkidul": (-7.9658, 110.6015),
    "Kulon Progo": (-7.8631, 110.1557),
    "Klaten": (-7.7058, 110.6009),
    "Magelang": (-7.4705, 110.2172),
    "Pasar Kembang": (-7.7916, 110.3648),
    "Kota Baru": (-7.7876, 110.3734),
    "Kronggahan": (-7.7400, 110.3400),
    "Muntilan": (-7.5815, 110.2796),
    "Solo Raya": (-7.5692, 110.8298),
    "Kadipaten": (-7.8032, 110.3574),
    "Maguwoharjo": (-7.7667, 110.4287),
    "Pringwulung": (-7.7656, 110.3867)
}

# ==============================================================================
# 1. FUNGSI LOGIKA & ALGORITMA
# ==============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius bumi (km)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_match(user_profile, candidates_df):
    results = []
    
    if len(candidates_df) == 0:
        return pd.DataFrame()

    # NLP Preprocessing
    all_essays = candidates_df['essay'].tolist() + [user_profile['essay']]
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_essays)
    except ValueError:
        return pd.DataFrame()
    
    user_vector = tfidf_matrix[-1]
    candidate_vectors = tfidf_matrix[:-1]

    for idx, row in candidates_df.iterrows():
        # --- LOGIKA FILTER (HARD FILTER) ---
        if user_profile['orientation'] == 'straight':
            if user_profile['sex'] == row['sex']: continue
        elif user_profile['orientation'] == 'gay':
            if user_profile['sex'] != row['sex']: continue
        
        # --- PERHITUNGAN SKOR & DETAIL ---
        scores = {}
        
        # 1. Usia (10%)
        age_diff = abs(user_profile['age'] - row['age'])
        score_age = 1.0 if age_diff <= 5 else max(0, 1.0 - (age_diff - 5)/10)
        scores['age'] = {'score': score_age, 'raw': f"Selisih {age_diff} tahun"}
        
        # 2. Gender (15%)
        # Convert 'm'/'f' to display text
        gender_display = "Male" if row['sex'] == 'm' else "Female"
        scores['sex'] = {'score': 1.0, 'raw': "Sesuai Preferensi"}

        # 3. Lokasi (20%)
        dist = haversine(user_profile['lat'], user_profile['lon'], row['lat'], row['lon'])
        score_loc = max(0, 1.0 - (dist / 50.0))
        scores['location'] = {'score': score_loc, 'raw': f"{round(dist, 1)} km"}
        
        # 4. Lifestyle (10%)
        life_score = 0
        match_list = []
        if user_profile['smokes'] == row['smokes']: 
            life_score += 0.5
            match_list.append("Rokok Sama")
        if user_profile['diet'] == row['diet']: 
            life_score += 0.5
            match_list.append("Diet Sama")
        if not match_list: match_list.append("Tidak ada yg sama")
        
        scores['lifestyle'] = {'score': life_score, 'raw': ", ".join(match_list)}

        # 5. Personality (25%)
        sim = cosine_similarity(user_vector, candidate_vectors[idx])
        score_nlp = sim[0][0]
        scores['personality'] = {'score': score_nlp, 'raw': f"Kemiripan {round(score_nlp*100)}%"}

        # Total Weighted Score
        final_score = (score_age*0.10) + (1.0*0.15) + (score_loc*0.20) + \
                      (life_score*0.10) + (score_nlp*0.25)
        
        results.append({
            'Nama': row['name'],
            'Umur': row['age'],
            'Gender': gender_display, # [UPDATE] Tampil sebagai Male/Female
            'Lokasi': row['location_name'],
            'Jarak (km)': round(dist, 1),
            'Skor Kecocokan': round(final_score * 100, 2),
            'raw_details': scores # Simpan detail untuk ditampilkan saat klik
        })
    
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values(by='Skor Kecocokan', ascending=False)

# ==============================================================================
# 2. SIDEBAR: PROFIL USER
# ==============================================================================
with st.sidebar:
    st.title("👤 Profil Anda")
    my_name = st.text_input("Nama Anda", "User Utama")
    my_age = st.number_input("Umur", 18, 99, 24)
    my_sex = st.selectbox("Gender", ["Pria", "Wanita"])
    my_orient = st.selectbox("Mencari...", ["Lawan Jenis (Straight)", "Sesama Jenis (Gay)", "Keduanya (Bisexual)"])
    
    sex_map = {"Pria": "m", "Wanita": "f"}
    orient_map = {"Lawan Jenis (Straight)": "straight", "Sesama Jenis (Gay)": "gay", "Keduanya (Bisexual)": "bisexual"}
    
    st.markdown("---")
    my_loc = st.selectbox("Lokasi Saat Ini", list(loc_options.keys()))
    my_lat, my_lon = loc_options[my_loc]
    my_smokes = st.selectbox("Merokok?", ["no", "sometimes", "yes"])
    my_diet = st.selectbox("Diet", ["anything", "vegetarian", "vegan"])
    st.markdown("---")
    my_essay = st.text_area("Essay", "I love coding, coffee, and sunset.")

    user_profile = {
        'name': my_name, 'age': my_age, 'sex': sex_map[my_sex], 
        'orientation': orient_map[my_orient], 'lat': my_lat, 'lon': my_lon,
        'smokes': my_smokes, 'diet': my_diet, 'essay': my_essay
    }

# ==============================================================================
# 3. MAIN CONTENT
# ==============================================================================
st.title("💘Matchmaker Dashboard (Jogja Edition)")

tab_input, tab_result = st.tabs(["📂 1. Input Data Kandidat", "📊 2. Hasil Rekomendasi"])

# --- TAB 1: INPUT DATA ---
with tab_input:
    col_input, col_table = st.columns([1, 2])
    
    with col_input:
        st.subheader("Tambah Kandidat")
        with st.form("add_form", clear_on_submit=True):
            c_name = st.text_input("Nama")
            c_age = st.number_input("Umur", 18, 99, 25)
            c_sex = st.selectbox("Gender", ["Pria", "Wanita"])
            c_loc = st.selectbox("Lokasi", list(loc_options.keys()))
            c_smokes = st.selectbox("Merokok", ["no", "sometimes", "yes"])
            c_diet = st.selectbox("Diet", ["anything", "vegetarian", "vegan"])
            c_essay = st.text_area("Essay", "Hobbies...")
            if st.form_submit_button("➕ Tambah"):
                st.session_state['candidates_data'].append({
                    'name': c_name, 'age': c_age, 'sex': sex_map[c_sex],
                    'location_name': c_loc, 'lat': loc_options[c_loc][0], 'lon': loc_options[c_loc][1],
                    'smokes': c_smokes, 'diet': c_diet, 'essay': c_essay
                })
                st.rerun()

    with col_table:
        st.subheader(f"Data Kandidat ({len(st.session_state['candidates_data'])})")
        if len(st.session_state['candidates_data']) > 0:
            st.dataframe(pd.DataFrame(st.session_state['candidates_data'])[['name', 'age', 'sex', 'location_name']], use_container_width=True)
            if st.button("🗑️ Reset Data"):
                st.session_state['candidates_data'] = []
                st.rerun()
        else:
            if st.button("📂 Load Dummy Data"):
                dummy_data = [
                    {'name': 'Cinta (24)', 'age': 24, 'sex': 'f', 'location_name': 'Jogja (Kota)', 'lat': -7.7956, 'lon': 110.3695, 'smokes': 'no', 'diet': 'anything', 'essay': 'I love hiking and outdoor adventures.'},
                    {'name': 'Rangga (30)', 'age': 30, 'sex': 'm', 'location_name': 'Klaten', 'lat': -7.7058, 'lon': 110.6009, 'smokes': 'no', 'diet': 'vegetarian', 'essay': 'Software engineer who loves coffee and coding.'},
                    {'name': 'Alya (22)', 'age': 22, 'sex': 'f', 'location_name': 'Sleman', 'lat': -7.7145, 'lon': 110.3456, 'smokes': 'sometimes', 'diet': 'vegan', 'essay': 'Artist and free spirit. Painting is my life.'},
                    {'name': 'Budi (35)', 'age': 35, 'sex': 'm', 'location_name': 'Bantul', 'lat': -7.8924, 'lon': 110.3323, 'smokes': 'yes', 'diet': 'anything', 'essay': 'Corporate professional. Golf and dining.'},
                    {'name': 'Siti (28)', 'age': 28, 'sex': 'f', 'location_name': 'Gunungkidul', 'lat': -7.9658, 'lon': 110.6015, 'smokes': 'no', 'diet': 'anything', 'essay': 'Doctor working late shifts. Looking for relaxation.'},
                    {'name': 'Santi (21)', 'age': 21, 'sex': 'f', 'location_name': 'Kulon Progo', 'lat': -7.8631, 'lon': 110.1557, 'smokes': 'no', 'diet': 'anything', 'essay': 'I love hiking and outdoor adventures.'},
                    {'name': 'Riko (30)', 'age': 30, 'sex': 'm', 'location_name': 'Magelang', 'lat': -7.4705, 'lon': 110.2172, 'smokes': 'no', 'diet': 'vegetarian', 'essay': 'Software engineer who loves coffee and coding.'},
                    {'name': 'Alin (20)', 'age': 20, 'sex': 'f', 'location_name': 'Pasar Kembang', 'lat': -7.7916, 'lon': 110.3648, 'smokes': 'sometimes', 'diet': 'vegan', 'essay': 'Artist and free spirit. Painting is my life.'},
                    {'name': 'Boy (22)', 'age': 22, 'sex': 'm', 'location_name': 'Kota Baru', 'lat': -7.7876, 'lon': 110.3734, 'smokes': 'yes', 'diet': 'anything', 'essay': 'Corporate professional. Golf and dining.'},
                    {'name': 'Susi (23)', 'age': 23, 'sex': 'f', 'location_name': 'Kronggahan', 'lat': -7.7400, 'lon': 110.3400, 'smokes': 'no', 'diet': 'anything', 'essay': 'Doctor working late shifts. Looking for relaxation.'},
                    {'name': 'Cika (27)', 'age': 27, 'sex': 'f', 'location_name': 'Muntilan', 'lat': -7.5815, 'lon': 110.2796, 'smokes': 'no', 'diet': 'anything', 'essay': 'I love hiking and outdoor adventures.'},
                    {'name': 'Roy (31)', 'age': 31, 'sex': 'm', 'location_name': 'Solo Raya', 'lat': -7.5692, 'lon': 110.8298, 'smokes': 'no', 'diet': 'vegetarian', 'essay': 'Software engineer who loves coffee and coding.'},
                    {'name': 'Ajeng (21)', 'age': 21, 'sex': 'f', 'location_name': 'Kadipaten', 'lat': -7.8032, 'lon': 110.3574, 'smokes': 'sometimes', 'diet': 'vegan', 'essay': 'Artist and free spirit. Painting is my life.'},
                    {'name': 'Bayu (33)', 'age': 33, 'sex': 'm', 'location_name': 'Maguwoharjo', 'lat': -7.7667, 'lon': 110.4287, 'smokes': 'yes', 'diet': 'anything', 'essay': 'Corporate professional. Golf and dining.'},
                    {'name': 'Cynthia (22)', 'age': 22, 'sex': 'f', 'location_name': 'Pringwulung', 'lat': -7.7656, 'lon': 110.3867, 'smokes': 'no', 'diet': 'anything', 'essay': 'Doctor working late shifts. Looking for relaxation.'},
                    {'name': 'Rosalia (27)', 'age': 27, 'sex': 'f', 'location_name': 'Muntilan', 'lat': -7.5815, 'lon': 110.2796, 'smokes': 'no', 'diet': 'anything', 'essay': 'I love hiking and outdoor adventures.'},
                    {'name': 'Prita (21)', 'age': 21, 'sex': 'f', 'location_name': 'Kadipaten', 'lat': -7.8032, 'lon': 110.3574, 'smokes': 'sometimes', 'diet': 'vegan', 'essay': 'Artist and free spirit. Painting is my life.'},
                    {'name': 'Orsy (22)', 'age': 22, 'sex': 'f', 'location_name': 'Pringwulung', 'lat': -7.7656, 'lon': 110.3867, 'smokes': 'no', 'diet': 'anything', 'essay': 'Looking for relaxation and good books.'},
                    {'name': 'Orys (23)', 'age': 23, 'sex': 'm', 'location_name': 'Kronggahan', 'lat': -7.7400, 'lon': 110.3400, 'smokes': 'no', 'diet': 'anything', 'essay': 'Gaming and technology enthusiast.'},
                    {'name': 'Sigit (31)', 'age': 31, 'sex': 'm', 'location_name': 'Solo Raya', 'lat': -7.5692, 'lon': 110.8298, 'smokes': 'no', 'diet': 'vegetarian', 'essay': 'Software engineer who loves coffee and coding.'},
                    {'name': 'Andriana (33)', 'age': 33, 'sex': 'f', 'location_name': 'Maguwoharjo', 'lat': -7.7667, 'lon': 110.4287, 'smokes': 'yes', 'diet': 'anything', 'essay': 'Corporate professional. Golf and dining.'}
                ]
                st.session_state['candidates_data'] = dummy_data
                st.rerun()

# --- TAB 2: HASIL ---
with tab_result:
    st.header("🔍 Analisis SPPK")
    
    df_candidates = pd.DataFrame(st.session_state['candidates_data'])
    
    if len(df_candidates) > 0:
        # Hitung Match
        final_df = calculate_match(user_profile, df_candidates)
        
        if not final_df.empty:
            # 1. Tampilkan Tabel Interaktif
            st.info("💡 **Klik salah satu baris** pada tabel di bawah untuk melihat Detail Penjelasan Skor.")
            
            # [UPDATE] Tabel dengan Gender Male/Female & Interaktif (Selection Mode)
            selection = st.dataframe(
                final_df[['Nama', 'Gender', 'Umur', 'Lokasi', 'Skor Kecocokan']],
                use_container_width=True,
                hide_index=True,
                on_select="rerun", # Mengaktifkan fitur klik baris
                selection_mode="single-row"
            )

            # 2. Logika Menampilkan Detail saat Baris Diklik
            if len(selection.selection['rows']) > 0:
                selected_idx = selection.selection['rows'][0]
                # Ambil data dari dataframe hasil berdasarkan index
                selected_row = final_df.iloc[selected_idx]
                details = selected_row['raw_details']
                
                st.divider()
                st.subheader(f"🕵️ Detail Analisis Skor: {selected_row['Nama']}")
                
                # Tampilan Kolom Breakdown
                c1, c2, c3, c4, c5 = st.columns(5)
                
                with c1:
                    st.metric("1. Usia (10%)", f"{round(details['age']['score']*100)}%", help="Semakin dekat umur, semakin tinggi.")
                    st.caption(f"Reason: {details['age']['raw']}")
                with c2:
                    st.metric("2. Gender (15%)", f"{round(details['sex']['score']*100)}%")
                    st.caption(f"Reason: {details['sex']['raw']}")
                with c3:
                    st.metric("3. Lokasi (20%)", f"{round(details['location']['score']*100)}%", help="Radius maks 50km.")
                    st.caption(f"Reason: {details['location']['raw']}")
                with c4:
                    st.metric("4. Lifestyle (10%)", f"{round(details['lifestyle']['score']*100)}%")
                    st.caption(f"Reason: {details['lifestyle']['raw']}")
                with c5:
                    st.metric("5. Essay (25%)", f"{round(details['personality']['score']*100)}%", help="Analisis kemiripan teks NLP.")
                    st.caption(f"Reason: {details['personality']['raw']}")
                
                # Visualisasi Radar Chart untuk kandidat terpilih
                categories = ['Usia', 'Gender', 'Lokasi', 'Lifestyle', 'Essay']
                values = [
                    details['age']['score'], details['sex']['score'], details['location']['score'], 
                    details['lifestyle']['score'], details['personality']['score']
                ]
                fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name=selected_row['Nama']))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), margin=dict(t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Tidak ditemukan kecocokan (Cek preferensi Gender Anda).")
    else:
        st.warning("Data kandidat kosong.")