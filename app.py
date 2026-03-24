import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.set_page_config(page_title="ADAD - Cartes", layout="wide")
st.title("🏥 Santé et Territoires : vulnérabilité en France")
st.markdown("Projet de Elisa Bon, Garance Pares et Elyne Cameriano")        

# Onglet CARTES

# ---------- Loaders ----------
@st.cache_data
def load_iris_shp():
    gdf = gpd.read_file("data/CONTOURS-IRIS_2018.shp")
    # code dept
    gdf["CODE_DEPT"] = gdf["INSEE_COM"].astype(str).str.zfill(5).str[:2]
    # fix invalid geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    # dissolve -> départements
    gdf_dept = gdf.dissolve(by="CODE_DEPT").reset_index()
    return gdf_dept

@st.cache_data
def load_isolement():
    df = pd.read_csv("data/insee_menage_isolement_dep.csv")
    return df

gdf_dept = load_iris_shp()
df_iso_dept = load_isolement()

# ---------- Map functions ----------
def make_map_isolement():
    gdf = gdf_dept.merge(df_iso_dept, left_on="CODE_DEPT", right_on="Departement", how="inner")
    

    fig, ax = plt.subplots(figsize=(15, 12))
    gdf.plot(
        column="TAUX_MENAGE_ISOLES_POND",
        cmap="YlOrRd",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Taux moyen d'isolement (%)", "loc": "lower left"},
    )
    ax.set_title("Vulnérabilité : Taux de Ménages Isolés par Département",
                 fontsize=18, fontweight="bold")
    ax.set_axis_off()
    return fig

@st.cache_data
def load_vuln():
    df = pd.read_csv("data/vulnerabilite_socioeco_dep.csv")
    return df

df_vuln = load_vuln()


def make_map_pauvrete():
    df_vuln = pd.read_csv("data/vulnerabilite_socioeco_dep.csv")
    gdf = gdf_dept.merge(df_vuln, left_on="CODE_DEPT", right_on="Departement", how="inner")
    fig, ax = plt.subplots(figsize=(15, 12))
    gdf.plot(
        column="TAUX_PAUVRETE_POND",
        cmap="YlOrRd",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Taux de pauvreté moyen (%)", "loc": "lower left"},
    )

    ax.set_title("Vulnérabilité : Taux de pauvreté par Département",
                 fontsize=18, fontweight="bold")
    ax.set_axis_off()
    return fig


def make_map_acces_soins():
    acces_medecin = pd.read_csv("data/acces_soins_dep.csv")

    
    gdf = gdf_dept.merge(acces_medecin, left_on="CODE_DEPT", right_on="departement", how="inner")

    fig, ax = plt.subplots(figsize=(15, 12))
    gdf.plot(
        column="APL_moyen",
        cmap="YlOrRd",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "APL moyen (pondéré)", "loc": "lower left"},
    )

    ax.set_title("Accès aux soins : APL moyen pondéré par Département",
                 fontsize=18, fontweight="bold")
    ax.set_axis_off()
    return fig

# --- Carte finale : Vulnérabilité totale ---

@st.cache_data
def load_isolement():
    df = pd.read_csv("data/insee_menage_isolement_dep.csv")
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df["Departement"] = df["Departement"].astype(str).str.zfill(2)
    return df

@st.cache_data
def load_vuln():
    df = pd.read_csv("data/vulnerabilite_socioeco_dep.csv")
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df["Departement"] = df["Departement"].astype(str).str.zfill(2)
    return df

@st.cache_data
def load_acces_soins():
    df = pd.read_csv("data/acces_soins_dep.csv")
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df["departement"] = df["departement"].astype(str).str.zfill(2)
    return df


def make_map_vulnerabilite_totale():
    df_pov = load_vuln()[["Departement", "TAUX_PAUVRETE_POND"]].copy()
    df_iso = load_isolement()[["Departement", "TAUX_MENAGE_ISOLES_POND"]].copy()

    df_apl = load_acces_soins()[["departement", "APL_moyen"]].copy()
    df_apl = df_apl.rename(columns={"departement": "Departement"})


    df_final = df_pov.merge(df_iso, on="Departement", how="inner").merge(df_apl, on="Departement", how="inner")


    for col in ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "APL_moyen"]:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")


    apl_min = df_final["APL_moyen"].min(skipna=True)
    apl_max = df_final["APL_moyen"].max(skipna=True)

    if pd.isna(apl_min) or pd.isna(apl_max) or apl_max == apl_min:
        st.error("Impossible de normaliser l'APL (min/max invalides).")
        return None

    df_final["VULN_APL_SCORE"] = 100 * (1 - (df_final["APL_moyen"] - apl_min) / (apl_max - apl_min))

    # --- Indice total (1/3 chacun) ---
    df_final["VULNERABILITE_TOTALE"] = (
        (1/3) * df_final["TAUX_PAUVRETE_POND"]
        + (1/3) * df_final["TAUX_MENAGE_ISOLES_POND"]
        + (1/3) * df_final["VULN_APL_SCORE"]
    )

    # --- Merge géométrie (gdf_dept a CODE_DEPT) ---
    gdf_map = gdf_dept.merge(df_final, left_on="CODE_DEPT", right_on="Departement", how="inner")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(15, 12))
    gdf_map.plot(
        column="VULNERABILITE_TOTALE",
        cmap="RdYlGn_r",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Indice de Vulnérabilité Totale", "loc": "lower left"},
    )

    ax.set_title(
        "Vulnérabilité Totale par Département\n(1/3 Pauvreté + 1/3 Isolement + 1/3 Désert Médical via APL)",
        fontsize=18, fontweight="bold", pad=20
    )
    ax.set_axis_off()

    top5 = df_final.sort_values("VULNERABILITE_TOTALE", ascending=False)[
        ["Departement", "VULNERABILITE_TOTALE"]
    ].head(5)

    return fig, top5



# Onglet CONTEXTE 
## Âge moyen par département :

@st.cache_data
def load_age_dep_from_iris():
    df = pd.read_csv("data/insee_population_age_iris.csv")

    # Sécuriser types
    for c in ["POP_TOTALE","POP_0_14","POP_15_29","POP_30_59","POP_60_74","POP_75_PLUS"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Code dept depuis IRIS (9 caractères -> 2 premiers = dept)
    df["CODE_IRIS"] = df["IRIS"].astype(str).str.zfill(9)
    df["CODE_DEPT"] = df["CODE_IRIS"].str[:2]

    # Milieux de classes (approximation)
    mid = {
        "POP_0_14": 7.0,
        "POP_15_29": 22.0,
        "POP_30_59": 44.5,
        "POP_60_74": 67.0,
        "POP_75_PLUS": 82.5,
    }

    # Âge moyen estimé par IRIS
    num = 0
    for col, m in mid.items():
        num = num + df[col] * m
    df["AGE_MOYEN_ESTIME_IRIS"] = num / df["POP_TOTALE"]

    # Agrégation département : moyenne pondérée par POP_TOTALE
    df = df.dropna(subset=["CODE_DEPT", "AGE_MOYEN_ESTIME_IRIS", "POP_TOTALE"])
    dep = df.groupby("CODE_DEPT").apply(
        lambda g: (g["AGE_MOYEN_ESTIME_IRIS"] * g["POP_TOTALE"]).sum() / g["POP_TOTALE"].sum()
    ).reset_index(name="AGE_MOYEN_DEP")

    return dep


def make_map_age_moyen():
    df_age_dep = load_age_dep_from_iris()

    gdf_map = gdf_dept.merge(df_age_dep, on="CODE_DEPT", how="inner")

    fig, ax = plt.subplots(figsize=(15, 12))
    gdf_map.plot(
        column="AGE_MOYEN_DEP",
        cmap="YlGnBu",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Âge moyen estimé", "loc": "lower left"},
    )

    ax.set_title("Contexte : Âge moyen estimé par département",
                 fontsize=18, fontweight="bold", pad=20)
    ax.set_axis_off()

    # ➜ Juste ça en plus
    top5 = df_age_dep.sort_values("AGE_MOYEN_DEP", ascending=False).head(5)

    return fig, top5


## Densité hospitalière :

@st.cache_data
def load_hopitaux_dep():
    df = pd.read_csv("data/hopitaux_clean_et_stats.csv")

    # On garde les lignes hôpital (au cas où il y a d'autres types)
    if "TYPE_LIGNE" in df.columns:
        df = df[df["TYPE_LIGNE"].astype(str).str.upper().str.contains("HOPITAL", na=False)].copy()

    # Construire CODE_DEPT :
    # 1) depuis CODE_POSTAL si présent
    df["CODE_POSTAL"] = pd.to_numeric(df.get("CODE_POSTAL"), errors="coerce")
    dept_from_cp = df["CODE_POSTAL"].dropna().astype(int).astype(str).str.zfill(5).str[:2]
    df.loc[dept_from_cp.index, "CODE_DEPT"] = dept_from_cp

    # 2) fallback depuis FINESS si CODE_POSTAL manquant
    df["FINESS"] = pd.to_numeric(df.get("FINESS"), errors="coerce")
    mask_missing = df["CODE_DEPT"].isna()
    dept_from_finess = (
        df.loc[mask_missing, "FINESS"]
        .dropna()
        .astype(int)
        .astype(str)
        .str.zfill(9)
        .str[:2]
    )
    df.loc[dept_from_finess.index, "CODE_DEPT"] = dept_from_finess

    # Compter le nombre d'hôpitaux par département
    hosp_dep = (
        df.dropna(subset=["CODE_DEPT"])
          .groupby("CODE_DEPT")
          .size()
          .reset_index(name="NB_HOPITAUX_DEPT")
    )

    return hosp_dep


def make_map_densite_hospitaliere():
    hosp_dep = load_hopitaux_dep()

    # Merge avec ta géométrie département (gdf_dept a CODE_DEPT)
    gdf_map = gdf_dept.merge(hosp_dep, on="CODE_DEPT", how="left")
    gdf_map["NB_HOPITAUX_DEPT"] = gdf_map["NB_HOPITAUX_DEPT"].fillna(0)

    fig, ax = plt.subplots(figsize=(15, 12))
    gdf_map.plot(
        column="NB_HOPITAUX_DEPT",
        cmap="YlOrRd",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Nombre d'hôpitaux", "loc": "lower left"},
    )

    ax.set_title("Contexte : densité hospitalière (nb d'hôpitaux) par département",
                 fontsize=18, fontweight="bold", pad=20)
    ax.set_axis_off()
    return fig


# Profil vulnérabilité
@st.cache_data
def get_df_vulnerabilite_dept():
    df_pov = load_vuln()[["Departement", "TAUX_PAUVRETE_POND"]].copy()
    df_iso = load_isolement()[["Departement", "TAUX_MENAGE_ISOLES_POND"]].copy()

    df_apl = load_acces_soins()[["departement", "APL_moyen"]].copy()
    df_apl = df_apl.rename(columns={"departement": "Departement"})

    df = df_pov.merge(df_iso, on="Departement", how="inner").merge(df_apl, on="Departement", how="inner")

    for col in ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "APL_moyen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalisation APL -> score vuln (0..100 inversé)
    apl_min = df["APL_moyen"].min(skipna=True)
    apl_max = df["APL_moyen"].max(skipna=True)

    df["VULN_APL_SCORE"] = 100 * (1 - (df["APL_moyen"] - apl_min) / (apl_max - apl_min))

    # Indice total 1/3-1/3-1/3
    df["VULNERABILITE_TOTALE"] = (
        (1/3) * df["TAUX_PAUVRETE_POND"]
        + (1/3) * df["TAUX_MENAGE_ISOLES_POND"]
        + (1/3) * df["VULN_APL_SCORE"]
    )

    # Indice social (utile pour “écart”)
    df["VULN_SOCIALE"] = 0.5 * (df["TAUX_PAUVRETE_POND"] + df["TAUX_MENAGE_ISOLES_POND"])

    return df

def make_map_ecart_social_soins():
    df = get_df_vulnerabilite_dept().copy()

    # Écart : social - soins (si positif => social plus élevé que vuln soins, et inversement)
    df["ECART_SOCIAL_SOINS"] = df["VULN_SOCIALE"] - df["VULN_APL_SCORE"]

    gdf_map = gdf_dept.merge(df[["Departement", "ECART_SOCIAL_SOINS"]],
                             left_on="CODE_DEPT", right_on="Departement", how="inner")

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf_map.plot(
    column="ECART_SOCIAL_SOINS",
    cmap="coolwarm",
    scheme="NaturalBreaks",
    k=6,
    edgecolor="black",
    linewidth=0.5,
    legend=True,
    ax=ax,
    legend_kwds={
        "title": "Écart (social - soins)",
        "fontsize": 8,
        "title_fontsize": 10,
        "loc": "center left",              # 👈 point d’ancrage
        "bbox_to_anchor": (1, 0.5),        # 👈 sort la légende à droite
    },
)
    ax.set_title("Écart vulnérabilité sociale vs accès aux soins\n(positif = social > soins)",
                 fontsize=18, fontweight="bold", pad=20)
    ax.set_axis_off()
    return fig

# =========================
# FONCTION PCA + CARTE
# =========================
def page_pca():
    st.title("Analyse PCA de la vulnérabilité")

    st.markdown("""
    ### Pour aller plus loin : pourquoi nous utilisons une ACP ?

    Au départ, l’indice global de vulnérabilité pouvait être être construit en donnant
    un poids égal à chaque dimension : pauvreté, isolement des ménages et accès aux soins.

    Cela revenait à attribuer **1/3 à chaque variable**, mais ce choix était arbitraire.

    Nous avons donc utilisé une **Analyse en Composantes Principales (ACP / PCA)**,
    une méthode statistique qui permet de construire un indice synthétique à partir
    des données elles-mêmes.

    **Pourquoi cette méthode ?**
    - elle évite de choisir les coefficients à la main
    - elle tient compte des relations entre les variables
    - elle identifie la combinaison de variables qui explique le mieux les écarts entre départements
    - elle permet donc de construire un **indice de vulnérabilité plus objectif**
    """)

    # Chargement des données déjà présentes dans ton projet
    df_pov = load_vuln()[["Departement", "TAUX_PAUVRETE_POND"]].copy()
    df_iso = load_isolement()[["Departement", "TAUX_MENAGE_ISOLES_POND"]].copy()
    df_apl = load_acces_soins()[["departement", "APL_moyen"]].copy()
    df_apl = df_apl.rename(columns={"departement": "Departement"})

    # Fusion
    df_final = df_pov.merge(df_iso, on="Departement", how="inner").merge(df_apl, on="Departement", how="inner")

    # Types numériques
    for col in ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "APL_moyen"]:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    df_final = df_final.dropna()

    # Inversion de l'accès aux soins
    df_final["MANQUE_ACCES_SOINS"] = df_final["APL_moyen"].max() - df_final["APL_moyen"]

    # Standardisation
    features = ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "MANQUE_ACCES_SOINS"]
    x = df_final[features].values
    x_scaled = StandardScaler().fit_transform(x)

    # PCA
    pca = PCA(n_components=1)
    vuln_pca = pca.fit_transform(x_scaled)
    # On inverse pas le signe pour avoir un score cohérent 
    df_final["VULNERABILITE_PCA"] = vuln_pca
    
    # Poids
    poids = pca.components_[0]
    indices_poids = pd.DataFrame({
        "Indicateur": features,
        "Poids_ACP": poids
    }).sort_values(by="Poids_ACP", ascending=False)

    # Score final 0-100
    v_min = df_final["VULNERABILITE_PCA"].min()
    v_max = df_final["VULNERABILITE_PCA"].max()
    df_final["SCORE_FINAL_0_100"] = 100 * (df_final["VULNERABILITE_PCA"] - v_min) / (v_max - v_min)

    st.subheader("Poids estimés par la PCA")
    st.dataframe(indices_poids, use_container_width=True)

    st.subheader("Top 5 des départements les plus vulnérables")
    top5 = df_final[["Departement", "SCORE_FINAL_0_100"]].sort_values(
        by="SCORE_FINAL_0_100", ascending=False
    ).head(5)
    st.dataframe(top5, use_container_width=True)

    st.subheader("Carte de la vulnérabilité totale par département")

    gdf_map = gdf_dept.merge(df_final, left_on="CODE_DEPT", right_on="Departement", how="inner")

    fig, ax = plt.subplots(figsize=(8, 6))
    gdf_map.plot(
        column="SCORE_FINAL_0_100",
        cmap="RdYlGn_r",
        scheme="NaturalBreaks",
        k=8,
        edgecolor="black",
        linewidth=0.5,
        legend=True,
        ax=ax,
        legend_kwds={"title": "Indice de Vulnérabilité Totale",    "fontsize": 8,
        "title_fontsize": 10,
        "loc": "center left",              # 👈 point d’ancrage
        "bbox_to_anchor": (1, 0.5),        # 👈 sort la légende à droite
    },
)
        

    ax.set_title(
        "Vulnérabilité Totale par Département - Indice construit par PCA",
        fontsize=18,
        fontweight="bold",
        pad=20
    )
    ax.set_axis_off()
    st.pyplot(fig)

    st.markdown("""
    ### Interprétation

    Cette carte représente un **indice global de vulnérabilité territoriale**
    construit à partir de trois dimensions :
    - la pauvreté
    - l’isolement
    - le manque d’accès aux soins

    Plus la couleur tend vers le **rouge**, plus le département cumule
    des facteurs défavorables.
    """)

# STREAMLIT

tab1, tab2, tab3, tab4 = st.tabs([ "Contexte", "Indice de Vulnérabilité", "Profil vulnérabilité", "PCA"])

with tab1:
    st.subheader("Contexte : âge moyen & densité hospitalière")
    st.markdown("#### Problématique : Comment identifier les territoires prioritaires pour l’action publique face au cumul de vulnérabilités ?")
    st.markdown("#### Objectif : Construire un indicateur synthétique permettant d’identifier les territoires les plus vulnérables en France ?")
    
    st.subheader("Indicateurs nationaux")

    # Chargement des données
    df_pov = load_vuln()[["Departement", "TAUX_PAUVRETE_POND"]].copy()
    df_iso = load_isolement()[["Departement", "TAUX_MENAGE_ISOLES_POND"]].copy()
    df_apl = load_acces_soins()[["departement", "APL_moyen"]].copy()
    df_apl = df_apl.rename(columns={"departement": "Departement"})

    # Fusion
    df = df_pov.merge(df_iso, on="Departement").merge(df_apl, on="Departement")

    # Conversion numérique
    for col in ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "APL_moyen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    # --- KPI 1 : pauvreté moyenne ---
    taux_pauvrete = df["TAUX_PAUVRETE_POND"].mean()

    # --- KPI 2 : isolement moyen ---
    taux_isolement = df["TAUX_MENAGE_ISOLES_POND"].mean()

    # --- KPI 3 : manque accès soins ---
    apl_min = df["APL_moyen"].min()
    apl_max = df["APL_moyen"].max()

    df["VULN_ACCES"] = 100 * (1 - (df["APL_moyen"] - apl_min) / (apl_max - apl_min))
    taux_acces = df["VULN_ACCES"].mean()

    # --- Affichage sur une ligne ---
    col1, col2, col3 = st.columns(3)

    col1.metric("Taux de pauvreté", f"{taux_pauvrete:.1f} %")
    col2.metric("Ménages isolés", f"{taux_isolement:.1f} %")
    col3.metric("Faible accès aux soins", f"{taux_acces:.1f} %")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Âge moyen par département")
        fig_age, top5_age = make_map_age_moyen()
        st.pyplot(fig_age, use_container_width=True)
        st.subheader("Top 5 départements les plus âgés")
        st.dataframe(top5_age, use_container_width=True)

    with colB:
        st.markdown("### Densité hospitalière")
        fig_hosp = make_map_densite_hospitaliere()
        st.pyplot(fig_hosp, use_container_width=True)


with tab2:
    st.subheader("Cartes vulnérabilité")
    # Création des 2 colonnes
    col_left, col_right = st.columns(2)

    # -------------------------
    # COLONNE GAUCHE
    # -------------------------
    with col_left:
        st.subheader("Sélection de la carte")

        choice = st.selectbox(
            "Choisis la carte à afficher :",
            [
                "1) Ménages isolés (département)",
                "2) Vulnérabilité socio-éco (département)",
                "3) Accès aux soins (département)",
            ],
        )

        if choice.startswith("1"):
            fig_left = make_map_isolement()
            st.pyplot(fig_left, use_container_width=True)

        elif choice.startswith("2"):
            fig_left = make_map_pauvrete()
            st.pyplot(fig_left, use_container_width=True)

        else:
            fig_left = make_map_acces_soins()
            st.pyplot(fig_left, use_container_width=True)

    # -------------------------
    # COLONNE DROITE
    # -------------------------
    with col_right:
        st.subheader("Carte indice vulnérabilité")

        res = make_map_vulnerabilite_totale()  # ou make_map_vulnerabilite_totale() selon ta fonction
        if res is not None:
            fig_right, top5_right = res
            st.pyplot(fig_right, use_container_width=True)

            st.subheader("Top 5 départements les plus vulnérables")
            st.dataframe(top5_right, use_container_width=True)

with tab3:
    st.subheader("Écart vulnérabilité sociale vs accès aux soins")
    fig = make_map_ecart_social_soins()
    st.pyplot(fig)
    st.subheader("Fiche département")

    df = get_df_vulnerabilite_dept().copy()
    dept_list = sorted(df["Departement"].dropna().unique().tolist())
    dept_choice = st.selectbox("Choisis un département (code)", dept_list)

    row = df[df["Departement"] == dept_choice].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vulnérabilité totale", f"{row['VULNERABILITE_TOTALE']:.1f}")
    c2.metric("Pauvreté (%)", f"{row['TAUX_PAUVRETE_POND']:.1f}")
    c3.metric("Isolement (%)", f"{row['TAUX_MENAGE_ISOLES_POND']:.1f}")
    c4.metric("Vuln soins (0-100)", f"{row['VULN_APL_SCORE']:.1f}")

    st.caption("Note : la vuln soins est une normalisation inversée de l’APL (plus l’accès est faible, plus le score est haut).")
    
    st.subheader("Analyse & interprétation")

    st.markdown("""
    ### Ce que montre l’indice
    - **La vulnérabilité totale** combine 3 dimensions : **pauvreté**, **isolement** et **accès aux soins**.
    - Deux territoires peuvent avoir le même score total mais **pour des raisons différentes** : c’est pour ça que les cartes par composante + la fiche département sont utiles.

    ### Comment lire l’écart “social – soins”
    - **Écart positif** : vulnérabilité sociale plus forte que la vuln soins.
    - **Écart négatif** : vuln soins plus forte (territoire plutôt “désert médical”).
    """)

    df = get_df_vulnerabilite_dept().copy()
    corr_cols = ["TAUX_PAUVRETE_POND", "TAUX_MENAGE_ISOLES_POND", "VULN_APL_SCORE", "VULNERABILITE_TOTALE"]
    corr = df[corr_cols].corr().round(2)

    
with tab4:
    page_pca()





