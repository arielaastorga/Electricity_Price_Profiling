from pathlib import Path
from datetime import date
import re



import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "raw"
MIN_AVAILABLE_DATE = date(2008, 1, 1)

st.set_page_config(page_title="Electricity Price Profiling", layout="wide")

st.markdown(
    """
    <style>
    .node-label {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.35rem;
    }
    .availability-note {
        font-size: 0.98rem;
        color: #374151;
        margin-top: 0.15rem;
        margin-bottom: 0.75rem;
    }
    .kpi-card {
        border-left: 4px solid #0f766e;
        background: #ffffff;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        min-height: 120px;
        margin-bottom: 0.8rem;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 800;
        color: #111827;
    }
    .kpi-sub {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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


def extract_month_from_filename(filename):
    match = re.match(r"^(\d{4})(\d{2})CmgBarras\.csv$", filename, flags=re.IGNORECASE)
    if not match:
        return None
    year, month = match.groups()
    return f"{year}-{month}"


def list_all_csv_files(data_dir=DATA_DIR):
    if not data_dir.exists():
        raise FileNotFoundError(f"La carpeta no existe: {data_dir}")
    return sorted([p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])


def get_files_for_months(months, data_dir=DATA_DIR):
    all_files = list_all_csv_files(data_dir)
    return [f for f in all_files if extract_month_from_filename(f.name) in months]


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
            df[col] = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_single_csv(file_path):
    expected_cols = ["fecha", "anio", "mes", "dia", "hora", "barra", "tension", "valor"]
    encodings_to_try = ["utf-8-sig", "utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, sep=";", encoding=enc, engine="python", skipinitialspace=True)
            df = clean_column_names(df)
            if all(col in df.columns for col in expected_cols):
                return convert_numeric_columns(df)
            last_error = f"Columnas detectadas en {file_path.name}: {list(df.columns)}"
        except Exception as e:
            last_error = e

    raise ValueError(f"No se pudo leer {file_path.name}. Último error: {last_error}")


@st.cache_data(show_spinner="Cargando archivos CSV...")
def load_selected_csvs(file_paths_as_str: tuple):
    dfs, loaded_files, failed_files = [], [], []
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

    return pd.concat(dfs, ignore_index=True), loaded_files, failed_files


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
        errors="coerce",
    )
    work["hora"] = pd.to_numeric(work["hora"], errors="coerce")
    work["valor"] = pd.to_numeric(work["valor"], errors="coerce")
    work = work.dropna(subset=["date", "hora", "valor"]).copy()
    work["hora"] = work["hora"].astype(int)

    # Mantengo 0-23 porque así quedó en la versión previa;
    # si tus datos vienen 1-24, esta parte conviene ajustarla.
    work = work[(work["hora"] >= 0) & (work["hora"] <= 23)].copy()

    if work.empty:
        raise ValueError("No hay datos válidos para construir las curvas.")

    work["day_type"] = work["date"].dt.dayofweek.map(lambda x: "Weekday" if x < 5 else "Weekend")
    return work.sort_values(["date", "hora"])


def filter_by_date_range(work: pd.DataFrame, start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return work[(work["date"] >= start_ts) & (work["date"] <= end_ts)].copy()


def format_number(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_kpis(work: pd.DataFrame):
    analysis = work.copy()
    analysis["year"] = analysis["date"].dt.year
    analysis["month_period"] = analysis["date"].dt.to_period("M")

    day_mask = analysis["hora"].between(7, 17)
    valley_mask = analysis["hora"].between(0, 6)
    peak_mask = analysis["hora"].between(18, 22)

    min_idx = analysis["valor"].idxmin()
    max_idx = analysis["valor"].idxmax()
    min_row = analysis.loc[min_idx]
    max_row = analysis.loc[max_idx]

    return {
        "avg_day_hours": analysis.loc[day_mask, "valor"].mean(),
        "avg_valley_hours": analysis.loc[valley_mask, "valor"].mean(),
        "avg_peak_hours": analysis.loc[peak_mask, "valor"].mean(),
        "min_value": min_row["valor"],
        "min_when": f"{min_row['date'].date()} - {int(min_row['hora']):02d}:00",
        "max_value": max_row["valor"],
        "max_when": f"{max_row['date'].date()} - {int(max_row['hora']):02d}:00",
        "daily_avg": analysis.groupby("date")["valor"].mean().mean(),
        "monthly_avg": analysis.groupby("month_period")["valor"].mean().mean(),
        "annual_avg": analysis.groupby("year")["valor"].mean().mean(),
        "n_days": int(analysis["date"].nunique()),
        "n_hours": int(len(analysis)),
    }


def cluster_profiles(work: pd.DataFrame, n_clusters: int, title: str):
    daily = (
        work.groupby(["date", "hora"])["valor"]
        .mean()
        .unstack(level="hora")
        .sort_index(axis=1)
    )

    all_hours = list(range(0, 24))
    daily = daily.reindex(columns=all_hours, fill_value=0)

    if daily.empty:
        raise ValueError("No hay días disponibles para generar perfiles.")

    if daily.shape[0] < n_clusters:
        raise ValueError(
            f"No hay suficientes días ({daily.shape[0]}) para formar {n_clusters} clusters."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(daily.values)

    model = KMedoids(
        n_clusters=n_clusters,
        metric="euclidean",
        init="k-medoids++",
        random_state=42,
    )
    labels = model.fit_predict(X_scaled)

    daily["cluster"] = labels
    medoid_positions = model.medoid_indices_
    medoid_dates = daily.index[medoid_positions]

    profiles = pd.DataFrame(daily.loc[medoid_dates, all_hours].values, columns=all_hours)
    profile_labels = []

    for cluster_id, medoid_date in enumerate(medoid_dates):
        freq = int((labels == cluster_id).sum())
        profile_labels.append(f"Cluster {cluster_id + 1} - {medoid_date.date()} ({freq} días)")

    profiles.index = profile_labels

    return {
        "profiles": profiles,
        "hours": all_hours,
        "daily_matrix": daily,
        "title": title,
    }


def build_profiles_plot(profiles_df: pd.DataFrame, hours: list[int], title: str):
    fig = go.Figure()

    for idx in profiles_df.index:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=profiles_df.loc[idx, hours].values,
                mode="lines+markers",
                name=str(idx),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour of the Day",
        yaxis_title="Marginal Price (USD/MWh)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def render_kpis(kpis):
    row1 = st.columns(3)
    with row1[0]:
        kpi_card("Average day hours (07-17h)", f"{format_number(kpis['avg_day_hours'])} USD/MWh")
    with row1[1]:
        kpi_card("Average peak hours (18-22h)", f"{format_number(kpis['avg_peak_hours'])} USD/MWh")
    with row1[2]:
        kpi_card("Average valley hours (00-06h)", f"{format_number(kpis['avg_valley_hours'])} USD/MWh")

    row2 = st.columns(3)
    with row2[0]:
        kpi_card("Minimum value", f"{format_number(kpis['min_value'])} USD/MWh", kpis["min_when"])
    with row2[1]:
        kpi_card("Maximum value", f"{format_number(kpis['max_value'])} USD/MWh", kpis["max_when"])
    with row2[2]:
        kpi_card("Daily average", f"{format_number(kpis['daily_avg'])} USD/MWh", f"{kpis['n_days']} días analizados")

    row3 = st.columns(2)
    with row3[0]:
        kpi_card("Monthly average", f"{format_number(kpis['monthly_avg'])} USD/MWh", f"{kpis['n_hours']} horas")
    with row3[1]:
        kpi_card("Annual average", f"{format_number(kpis['annual_avg'])} USD/MWh", "Promedio sobre los años incluidos")


def main():
    st.title("Electricity Price Profiling")
    st.write(
        "This application retrieves, processes, and visualizes hourly marginal electricity prices "
        "by grid node from the Chilean National Electric Coordinator (CEN)."
    )

    barra_options = []
    try:
        all_files = list_all_csv_files(DATA_DIR)
        if all_files:
            preview_df, _, _ = load_selected_csvs(tuple(str(p) for p in all_files[:1]))
            if "barra" in preview_df.columns:
                barra_options = sorted(preview_df["barra"].dropna().astype(str).unique().tolist())
    except Exception:
        barra_options = []

    st.markdown('<div class="node-label">Select a node</div>', unsafe_allow_html=True)
    if barra_options:
        selected_barra = st.selectbox("", options=barra_options, label_visibility="collapsed")
    else:
        selected_barra = st.text_input("", value="", placeholder="Type a node name", label_visibility="collapsed")
        if not selected_barra:
            st.info("No se pudieron precargar las barras. Escribe el nombre exacto de la barra.")

    st.markdown(
        f'<div class="availability-note">Available data from {MIN_AVAILABLE_DATE.strftime("%d-%m-%Y")} to today.</div>',
        unsafe_allow_html=True,
    )

    selected_range = st.date_input(
        "Select date range",
        value=(date(2024, 1, 1), date(2024, 12, 31)),
        min_value=MIN_AVAILABLE_DATE,
        max_value=date.today(),
    )

    try:
        start_date, end_date = selected_range
    except (TypeError, ValueError):
        st.warning("Please select both a start date and an end date.")
        st.stop()

    if start_date < MIN_AVAILABLE_DATE or end_date > date.today():
        st.error("El rango seleccionado está fuera de la ventana de datos disponible.")
        st.stop()

    if months_between(start_date, end_date) >= 12:
        st.error("Ingrese un período menor a 12 meses.")
        st.stop()

    months = get_required_months(start_date, end_date)

    try:
        selected_files = get_files_for_months(months, data_dir=DATA_DIR)
        if not selected_files:
            st.error("No se encontraron archivos CSV para el rango seleccionado.")
            st.stop()

        df, loaded_files, failed_files = load_selected_csvs(tuple(str(p) for p in selected_files))
        work = prepare_work_df(df, barra=selected_barra)
        work = filter_by_date_range(work, start_date, end_date)

        if work.empty:
            st.error("No hay datos disponibles para la barra y período seleccionados.")
            st.stop()

    except Exception as e:
        st.error(f"Error while loading files: {e}")
        st.stop()

    st.subheader("Statistical summary")
    render_kpis(compute_kpis(work))

    colw, colf, colt = st.columns(3)
    with colw:
        weekday_k = st.slider("Weekday clusters (Mon-Fri)", min_value=1, max_value=4, value=2, step=1)
    with colf:
        weekend_k = st.slider("Weekend clusters (Sat-Sun)", min_value=1, max_value=4, value=1, step=1)
    with colt:
        total_k = st.slider("All-days clusters", min_value=1, max_value=4, value=2, step=1)

    if st.button("Show representative profiles", use_container_width=True):
        try:
            weekday_work = work[work["day_type"] == "Weekday"].copy()
            weekend_work = work[work["day_type"] == "Weekend"].copy()

            st.session_state["weekday_profiles"] = cluster_profiles(
                weekday_work,
                weekday_k,
                "Representative weekday profiles (K-Medoids)"
            )
            st.session_state["weekend_profiles"] = cluster_profiles(
                weekend_work,
                weekend_k,
                "Representative weekend profiles (K-Medoids)"
            )
            st.session_state["all_profiles"] = cluster_profiles(
                work,
                total_k,
                "Representative profiles for all days (K-Medoids)"
            )

            st.success("Representative K-Medoids profiles generated successfully.")

        except Exception as e:
            st.error(f"Error while generating representative profiles: {e}")

    if "weekday_profiles" in st.session_state:
        result = st.session_state["weekday_profiles"]
        st.subheader("Weekday representative profiles (K-Medoids)")
        st.plotly_chart(
            build_profiles_plot(result["profiles"], result["hours"], result["title"]),
            use_container_width=True
        )
        with st.expander("Weekday representative data"):
            st.dataframe(result["profiles"], use_container_width=True)

    if "weekend_profiles" in st.session_state:
        result = st.session_state["weekend_profiles"]
        st.subheader("Weekend representative profiles (K-Medoids)")
        st.plotly_chart(
            build_profiles_plot(result["profiles"], result["hours"], result["title"]),
            use_container_width=True
        )
        with st.expander("Weekend representative data"):
            st.dataframe(result["profiles"], use_container_width=True)

    if "all_profiles" in st.session_state:
        result = st.session_state["all_profiles"]
        st.subheader("Representative profiles for all days (K-Medoids)")
        st.plotly_chart(
            build_profiles_plot(result["profiles"], result["hours"], result["title"]),
            use_container_width=True
        )
        with st.expander("All-days representative data"):
            st.dataframe(result["profiles"], use_container_width=True)

    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", work.shape[0])
    c2.metric("Analyzed days", work["date"].nunique())
    c3.metric("Loaded files", len(loaded_files))

    if failed_files:
        with st.expander("Files with errors"):
            failed_df = pd.DataFrame(failed_files, columns=["file", "error"])
            st.dataframe(failed_df, use_container_width=True)

    st.subheader("Preview")
    st.dataframe(work.head(20), use_container_width=True)


if __name__ == "__main__":
    main()