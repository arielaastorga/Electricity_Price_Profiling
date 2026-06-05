# streamlit_app.py
# Versión adaptada para Streamlit Community Cloud

from pathlib import Path
from datetime import date
import re

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIG
# ==========================================================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "raw"

st.set_page_config(
    page_title="Electricity Price Profiling",
    layout="wide"
)


# ==========================================================
# FECHAS
# ==========================================================
def get_required_months(start_date, end_date):
    if start_date > end_date:
        raise ValueError("La fecha inicial no puede ser mayor que la fecha final.")

    months = []
    year = start_date.year
    month = start_date.month

    while (year, month) <= (end_date.year, end_date.month):
        months.append(f"{year:04d}-{month:02d}")

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return months


def months_between(start_date, end_date):
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)


# ==========================================================
# ARCHIVOS
# ==========================================================
def extract_month_from_filename(filename):
    match = re.match(r"^(\d{4})(\d{2})CmgBarras\.csv$", filename, flags=re.IGNORECASE)
    if not match:
        return None
    year, month = match.groups()
    return f"{year}-{month}"


def list_all_csv_files(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"La carpeta no existe: {data_dir}")

    return sorted(
        [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    )


def get_files_for_months(months, data_dir=DATA_DIR):
    all_files = list_all_csv_files(data_dir)
    return [f for f in all_files if extract_month_from_filename(f.name) in months]


# ==========================================================
# LECTURA CSV
# ==========================================================
def clean_column_names(df):
    cleaned = []
    for c in df.columns:
        col = str(c).strip().lower()
        col = col.replace("\ufeff", "")
        col = col.replace('"', "")
        col = col.replace("'", "")
        cleaned.append(col)
    df.columns = cleaned
    return df


def convert_numeric_columns(df):
    numeric_cols = ["anio", "mes", "dia", "hora", "tension", "valor"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_single_csv(file_path):
    expected_cols = ["fecha", "anio", "mes", "dia", "hora", "barra", "tension", "valor"]
    encodings_to_try = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                file_path,
                sep=";",
                encoding=enc,
                engine="python",
                skipinitialspace=True
            )

            df = clean_column_names(df)

            if all(col in df.columns for col in expected_cols):
                df = convert_numeric_columns(df)
                return df

            last_error = f"Columnas detectadas en {file_path.name}: {list(df.columns)}"

        except Exception as e:
            last_error = e

    raise ValueError(f"No se pudo leer {file_path.name}. Último error: {last_error}")


@st.cache_data(show_spinner="Cargando archivos CSV...")
def load_selected_csvs(file_paths_as_str: tuple):
    dfs = []
    loaded_files = []
    failed_files = []

    for file_str in file_paths_as_str:
        file = Path(file_str)
        try:
            df = read_single_csv(file)
            df["source_file"] = file.name
            dfs.append(df)
            loaded_files.append(file.name)
        except Exception as e:
            failed_files.append((file.name, str(e)))

    if not dfs:
        raise ValueError("No se pudo leer ningún archivo válido.")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, loaded_files, failed_files


# ==========================================================
# PREPARACIÓN
# ==========================================================
def prepare_work_df(df: pd.DataFrame, barra: str | None = None):
    required_cols = ["anio", "mes", "dia", "hora", "valor"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    work = df.copy()

    if barra is not None and "barra" in work.columns:
        work = work[work["barra"].astype(str) == str(barra)].copy()

    if work.empty:
        raise ValueError("No hay datos disponibles después del filtro aplicado.")

    work["date"] = pd.to_datetime(
        {
            "year": pd.to_numeric(work["anio"], errors="coerce"),
            "month": pd.to_numeric(work["mes"], errors="coerce"),
            "day": pd.to_numeric(work["dia"], errors="coerce"),
        },
        errors="coerce"
    )

    work["hora"] = pd.to_numeric(work["hora"], errors="coerce")
    work["valor"] = pd.to_numeric(work["valor"], errors="coerce")

    work = work.dropna(subset=["date", "hora", "valor"]).copy()
    work["hora"] = work["hora"].astype(int)
    work = work[(work["hora"] >= 1) & (work["hora"] <= 24)].copy()

    if work.empty:
        raise ValueError("No hay datos válidos para construir las curvas.")

    return work.sort_values(["date", "hora"])


def filter_by_date_range(work: pd.DataFrame, start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return work[(work["date"] >= start_ts) & (work["date"] <= end_ts)].copy()


# ==========================================================
# KMEANS CURVES
# ==========================================================
def kmeans_curves(df: pd.DataFrame, n_clusters: int = 7, barra: str | None = None,
                  start_date=None, end_date=None):
    work = prepare_work_df(df, barra=barra)

    if start_date is not None and end_date is not None:
        work = filter_by_date_range(work, start_date, end_date)

    if work.empty:
        raise ValueError("No hay datos dentro del rango seleccionado.")

    daily = (
        work.groupby(["date", "hora"])["valor"]
        .mean()
        .unstack(level="hora")
        .fillna(0)
        .sort_index(axis=1)
    )

    all_hours = list(range(1, 25))
    daily = daily.reindex(columns=all_hours, fill_value=0)

    if daily.shape[0] < n_clusters:
        raise ValueError(
            f"No hay suficientes días ({daily.shape[0]}) para formar {n_clusters} clusters."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily.values)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    daily["cluster"] = model.fit_predict(X_scaled)
    daily["mes"] = daily.index.month

    return {
        "daily_matrix": daily,
        "hours": all_hours,
        "model": model,
        "scaler": scaler,
    }


def build_typical_profiles_for_period(daily_results: dict):
    daily = daily_results["daily_matrix"].copy()
    hours = daily_results["hours"]

    cluster_counts = daily["cluster"].value_counts()
    dominant_cluster = cluster_counts.idxmax()

    perfiles_df = pd.DataFrame([
        daily[daily["cluster"] == dominant_cluster][hours].mean()
    ])
    perfiles_df.index = ["Selected period"]

    return {
        "cluster_reference": {"Selected period": int(dominant_cluster)},
        "profiles": perfiles_df,
        "hours": hours,
        "mode": "period"
    }


def build_monthly_typical_profiles(daily_results: dict):
    daily = daily_results["daily_matrix"].copy()
    hours = daily_results["hours"]

    mes_cluster = (
        daily.groupby("mes")["cluster"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    perfiles = {}
    for mes, clus in mes_cluster.items():
        perfil = daily[daily["cluster"] == clus][hours].mean()
        perfiles[mes] = perfil

    perfiles_df = pd.DataFrame(perfiles).T
    perfiles_df.index.name = "month"
    perfiles_df = perfiles_df.sort_index()

    return {
        "cluster_reference": mes_cluster,
        "profiles": perfiles_df,
        "hours": hours,
        "mode": "monthly"
    }


def build_single_day_curve(df: pd.DataFrame, barra: str | None, target_date):
    work = prepare_work_df(df, barra=barra)
    target_ts = pd.Timestamp(target_date)

    work = work[work["date"] == target_ts].copy()

    if work.empty:
        raise ValueError("No data available for the selected day and node.")

    curve = (
        work.groupby("hora")["valor"]
        .mean()
        .sort_index()
        .reindex(range(1, 25), fill_value=0)
    )

    curve_df = pd.DataFrame(
        [curve.values],
        index=[str(target_ts.date())],
        columns=list(range(1, 25))
    )

    return {
        "profiles": curve_df,
        "hours": list(range(1, 25)),
        "mode": "single_day"
    }


# ==========================================================
# PLOT
# ==========================================================
def build_profiles_plot(profiles_df: pd.DataFrame, hours: list[int], title: str):
    fig = go.Figure()

    for idx in profiles_df.index:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=profiles_df.loc[idx, hours].values,
                mode="lines+markers",
                name=str(idx)
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Marginal Price (USD/MWh)",
        template="plotly_white",
        hovermode="x unified"
    )

    return fig


# ==========================================================
# APP
# ==========================================================
def main():
    st.title("Electricity Price Profiling")
    st.write(
        "This application retrieves, processes, and visualizes hourly marginal electricity prices "
        "by grid node from the Chilean National Electric Coordinator (CEN), enabling exploratory "
        "analysis of daily, short-term, and monthly price profiles."
    )

    selected_range = st.date_input(
        "Select date range",
        value=(date(2024, 1, 1), date(2024, 12, 31))
    )

    try:
        start_date, end_date = selected_range
    except (TypeError, ValueError):
        st.warning("Please select both a start date and an end date.")
        st.stop()

    st.info(f"Select a start date from 01-2008")

    try:
        months = get_required_months(start_date, end_date)
        selected_files = get_files_for_months(months, data_dir=DATA_DIR)

        if not selected_files:
            st.warning("No CSV files were found for the selected date range.")
            st.stop()

        df, loaded_files, failed_files = load_selected_csvs(
            tuple(str(p) for p in selected_files)
        )

        st.session_state["df"] = df
        st.session_state["selected_files"] = loaded_files
        st.session_state["failed_files"] = failed_files

    except Exception as e:
        st.error(f"Error while loading files: {e}")
        st.stop()

    df = st.session_state["df"]
    loaded_files = st.session_state.get("selected_files", [])
    failed_files = st.session_state.get("failed_files", [])

    with st.expander(f"Loaded files ({len(loaded_files)})"):
        st.write(loaded_files)

    if failed_files:
        with st.expander("Files with errors"):
            failed_df = pd.DataFrame(failed_files, columns=["file", "error"])
            st.dataframe(failed_df, use_container_width=True)

    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Loaded files", len(loaded_files))

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    barra_options = []
    if "barra" in df.columns:
        barra_options = sorted(df["barra"].dropna().astype(str).unique().tolist())

    selected_barra = None
    if barra_options:
        selected_barra = st.selectbox(
            "Select a node",
            options=["All"] + barra_options,
            index=0
        )
        if selected_barra == "All":
            selected_barra = None

    n_clusters = st.slider(
        "Number of clusters",
        min_value=2,
        max_value=12,
        value=7,
        step=1
    )

    if st.button("Show profiles", use_container_width=True):
        try:
            day_diff = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
            month_diff = months_between(start_date, end_date)

            if day_diff == 0:
                result = build_single_day_curve(
                    df=df,
                    barra=selected_barra,
                    target_date=start_date
                )
                result["title"] = f"Hourly curve - {start_date}"
                st.session_state["curve_results"] = result

            elif month_diff == 0:
                daily_results = kmeans_curves(
                    df=df,
                    n_clusters=n_clusters,
                    barra=selected_barra,
                    start_date=start_date,
                    end_date=end_date
                )
                result = build_typical_profiles_for_period(daily_results)
                result["title"] = "Typical profiles for the selected period (KMeans)"
                result["daily_matrix"] = daily_results["daily_matrix"]
                st.session_state["curve_results"] = result

            else:
                daily_results = kmeans_curves(
                    df=df,
                    n_clusters=n_clusters,
                    barra=selected_barra,
                    start_date=start_date,
                    end_date=end_date
                )
                result = build_monthly_typical_profiles(daily_results)
                result["title"] = "Monthly typical day profiles (KMeans)"
                result["daily_matrix"] = daily_results["daily_matrix"]
                st.session_state["curve_results"] = result

            st.success("Profiles generated successfully.")

        except Exception as e:
            st.error(f"Error while generating profiles: {e}")

    if "curve_results" in st.session_state:
        result = st.session_state["curve_results"]
        profiles = result["profiles"]
        hours = result["hours"]
        title = result["title"]
        mode = result["mode"]

        st.subheader("Profiles")
        fig = build_profiles_plot(profiles, hours, title)
        st.plotly_chart(fig, use_container_width=True)

        if "daily_matrix" in result:
            with st.expander("Daily matrix"):
                st.dataframe(result["daily_matrix"].head(30), use_container_width=True)

        if mode == "monthly":
            st.subheader("Dominant cluster by month")
            cluster_df = pd.DataFrame(
                [{"month": k, "dominant_cluster": v} for k, v in result["cluster_reference"].items()]
            ).sort_values("month")
            st.dataframe(cluster_df, use_container_width=True)

        elif mode == "period":
            st.subheader("Dominant cluster for the selected period")
            cluster_df = pd.DataFrame(
                [{"period": k, "dominant_cluster": v} for k, v in result["cluster_reference"].items()]
            )
            st.dataframe(cluster_df, use_container_width=True)

        st.subheader("Plotted data")
        st.dataframe(profiles, use_container_width=True)

        csv_profiles = profiles.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="Download profiles",
            data=csv_profiles,
            file_name="profiles_output.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()