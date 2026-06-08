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

SOLAR_BORDER = "#1d4ed8"
CURTAILMENT_BORDER = "#7dd3fc"
GENERAL_BORDER = "#0f766e"

st.set_page_config(page_title="Electricity Price Profiling", layout="wide")

st.markdown(
    f"""
    <style>
    .field-label {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }}
    .field-note {{
        font-size: 0.95rem;
        font-weight: 400;
        color: #4b5563;
        margin-top: 0rem;
        margin-bottom: 0.7rem;
    }}
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }}
    .kpi-card {{
        background: #ffffff;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        min-height: 120px;
        margin-bottom: 0.8rem;
    }}
    .kpi-title {{
        font-size: 0.82rem;
        font-weight: 700;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.9rem;
        line-height: 1.1;
        font-weight: 800;
        color: #111827;
    }}
    .kpi-sub {{
        font-size: 0.88rem;
        color: #6b7280;
        margin-top: 0.35rem;
    }}
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


def get_month_starts(start_date, end_date):
    month_starts = []
    current = pd.Timestamp(start_date).replace(day=1)

    while current <= pd.Timestamp(end_date):
        month_starts.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    return month_starts


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
                df = convert_numeric_columns(df)
                return df
            last_error = f"Columnas detectadas en {file_path.name}: {list(df.columns)}"
        except Exception as e:
            last_error = e

    raise ValueError(f"No se pudo leer {file_path.name}. Último error: {last_error}")


@st.cache_data(show_spinner="Loading CSV files...")
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

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(how="all")
    return combined, loaded_files, failed_files


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

    work = work.dropna(subset=["date", "hora"]).copy()
    work["hora"] = work["hora"].astype(int)
    work = work[(work["hora"] >= 1) & (work["hora"] <= 24)].copy()

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


def format_int(value):
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}".replace(",", ".")


def kpi_card(title, value, subtitle="", border_color=GENERAL_BORDER):
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 4px solid {border_color};">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compute_kpis(work: pd.DataFrame):
    analysis = work.dropna(subset=["valor"]).copy()

    if analysis.empty:
        raise ValueError("No hay valores válidos para calcular KPIs.")

    solar_mask = analysis["hora"].between(8, 18)
    non_solar_mask = analysis["hora"].between(19, 24) | analysis["hora"].between(1, 7)

    min_idx = analysis["valor"].idxmin()
    max_idx = analysis["valor"].idxmax()
    min_row = analysis.loc[min_idx]
    max_row = analysis.loc[max_idx]

    total_hours = len(analysis)

    cmg_eq_0 = (analysis["valor"] == 0).sum()
    cmg_lt_30 = (analysis["valor"] < 30).sum()
    cmg_lt_50 = (analysis["valor"] < 50).sum()
    cmg_lt_100 = (analysis["valor"] < 100).sum()

    avg_period = analysis["valor"].mean()
    solar_avg = analysis.loc[solar_mask, "valor"].mean()
    non_solar_avg = analysis.loc[non_solar_mask, "valor"].mean()

    return {
        "avg_period": avg_period,
        "p50": analysis["valor"].quantile(0.50),
        "p90": analysis["valor"].quantile(0.90),
        "min_value": min_row["valor"],
        "min_when": f"{min_row['date'].date()} - {int(min_row['hora']):02d}:00",
        "max_value": max_row["valor"],
        "max_when": f"{max_row['date'].date()} - {int(max_row['hora']):02d}:00",
        "solar_avg": solar_avg,
        "non_solar_avg": non_solar_avg,
        "spread_solar_non_solar": non_solar_avg - solar_avg if pd.notna(solar_avg) and pd.notna(non_solar_avg) else pd.NA,
        "cmg_eq_0": cmg_eq_0,
        "cmg_lt_30": cmg_lt_30,
        "cmg_lt_50": cmg_lt_50,
        "cmg_lt_100": cmg_lt_100,
        "cmg_eq_0_pct": (cmg_eq_0 / total_hours * 100) if total_hours else pd.NA,
        "cmg_lt_30_pct": (cmg_lt_30 / total_hours * 100) if total_hours else pd.NA,
        "cmg_lt_50_pct": (cmg_lt_50 / total_hours * 100) if total_hours else pd.NA,
        "cmg_lt_100_pct": (cmg_lt_100 / total_hours * 100) if total_hours else pd.NA,
        "n_days": int(analysis["date"].nunique()),
        "n_hours": int(total_hours),
    }


def build_daily_matrix_with_imputation(work: pd.DataFrame):
    subset = work.copy()
    subset = subset.groupby(["date", "hora"], as_index=False)["valor"].mean()

    daily = (
        subset.pivot(index="date", columns="hora", values="valor")
        .sort_index()
        .sort_index(axis=1)
    )

    all_hours = list(range(1, 25))
    daily = daily.reindex(columns=all_hours)

    daily = daily.ffill(axis=0)
    daily = daily.ffill(axis=1)
    daily = daily.bfill(axis=1)
    daily = daily.bfill(axis=0)

    return daily, all_hours


def cluster_profiles(work: pd.DataFrame, n_clusters: int, title: str):
    daily, all_hours = build_daily_matrix_with_imputation(work)

    if daily.empty:
        raise ValueError("No hay días disponibles para generar perfiles.")

    if daily.isna().any().any():
        raise ValueError("Persisten NaN después de la imputación.")

    if daily.shape[0] < n_clusters:
        raise ValueError(
            f"No hay suficientes días completos ({daily.shape[0]}) para formar {n_clusters} clusters."
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

    daily_labeled = daily.copy()
    daily_labeled["cluster"] = labels
    medoid_positions = model.medoid_indices_
    medoid_dates = daily_labeled.index[medoid_positions]

    profiles = pd.DataFrame(daily_labeled.loc[medoid_dates, all_hours].values, columns=all_hours)
    profile_labels = []

    for cluster_id, medoid_date in enumerate(medoid_dates):
        freq = int((labels == cluster_id).sum())
        profile_labels.append(f"Cluster {cluster_id + 1} - {medoid_date.date()} ({freq} days)")

    profiles.index = profile_labels

    return {
        "profiles": profiles,
        "hours": all_hours,
        "title": title,
    }


def build_single_day_curve(work: pd.DataFrame, target_date):
    target_ts = pd.Timestamp(target_date)
    one_day = work[work["date"] == target_ts].copy()

    if one_day.empty:
        raise ValueError("No data available for the selected day and node.")

    curve = (
        one_day.groupby("hora")["valor"]
        .mean()
        .sort_index()
        .reindex(range(1, 25))
    )

    curve = curve.ffill().bfill()

    if curve.isna().any():
        raise ValueError("No fue posible imputar completamente la curva diaria.")

    curve_df = pd.DataFrame(
        [curve.values],
        index=[str(target_ts.date())],
        columns=list(range(1, 25)),
    )

    return {
        "profiles": curve_df,
        "hours": list(range(1, 25)),
        "title": f"Hourly curve - {target_ts.date()}",
    }


def build_monthly_representative_profiles(work: pd.DataFrame, start_date, end_date, title: str):
    daily, all_hours = build_daily_matrix_with_imputation(work)

    if daily.empty:
        raise ValueError("No hay días disponibles para construir perfiles mensuales.")

    if daily.isna().any().any():
        raise ValueError("Persisten NaN después de la imputación mensual.")

    month_starts = get_month_starts(start_date, end_date)
    monthly_profiles = {}

    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        month_mask = (daily.index >= month_start) & (daily.index <= month_end)
        month_daily = daily.loc[month_mask].copy()

        if month_daily.empty:
            continue

        representative_curve = month_daily.mean(axis=0)
        month_label = f"Month {month_start.month}"
        monthly_profiles[month_label] = representative_curve

    if not monthly_profiles:
        raise ValueError("No se pudieron construir curvas mensuales para el rango seleccionado.")

    profiles_df = pd.DataFrame(monthly_profiles).T
    profiles_df = profiles_df[all_hours]

    return {
        "profiles": profiles_df,
        "hours": all_hours,
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
    st.markdown('<div class="section-title">General KPI</div>', unsafe_allow_html=True)
    row1 = st.columns(3)
    with row1[0]:
        kpi_card(
            "Selected period average",
            f"{format_number(kpis['avg_period'])} USD/MWh",
            f"{kpis['n_days']} days · {kpis['n_hours']} hours",
            border_color=GENERAL_BORDER,
        )
    with row1[1]:
        kpi_card("P50", f"{format_number(kpis['p50'])} USD/MWh", border_color=GENERAL_BORDER)
    with row1[2]:
        kpi_card("P90", f"{format_number(kpis['p90'])} USD/MWh", border_color=GENERAL_BORDER)

    row2 = st.columns(2)
    with row2[0]:
        kpi_card("Minimum value", f"{format_number(kpis['min_value'])} USD/MWh", kpis["min_when"], border_color=GENERAL_BORDER)
    with row2[1]:
        kpi_card("Maximum value", f"{format_number(kpis['max_value'])} USD/MWh", kpis["max_when"], border_color=GENERAL_BORDER)

    st.markdown('<div class="section-title">Solar window KPI</div>', unsafe_allow_html=True)
    row3 = st.columns(3)
    with row3[0]:
        kpi_card("Average solar hours (08h-18h)", f"{format_number(kpis['solar_avg'])} USD/MWh", border_color=SOLAR_BORDER)
    with row3[1]:
        kpi_card("Average non-solar hours (19h-07h)", f"{format_number(kpis['non_solar_avg'])} USD/MWh", border_color=SOLAR_BORDER)
    with row3[2]:
        kpi_card("Solar / non-solar spread", f"{format_number(kpis['spread_solar_non_solar'])} USD/MWh", "Non-solar minus solar", border_color=SOLAR_BORDER)

    st.markdown('<div class="section-title">Curtailment</div>', unsafe_allow_html=True)
    row4 = st.columns(4)
    with row4[0]:
        kpi_card(
            "Hours with CMG = 0",
            f"{format_int(kpis['cmg_eq_0'])} h",
            f"{format_number(kpis['cmg_eq_0_pct'])}% of sample",
            border_color=CURTAILMENT_BORDER,
        )
    with row4[1]:
        kpi_card(
            "Hours with CMG < 30",
            f"{format_int(kpis['cmg_lt_30'])} h",
            f"{format_number(kpis['cmg_lt_30_pct'])}% of sample",
            border_color=CURTAILMENT_BORDER,
        )
    with row4[2]:
        kpi_card(
            "Hours with CMG < 50",
            f"{format_int(kpis['cmg_lt_50'])} h",
            f"{format_number(kpis['cmg_lt_50_pct'])}% of sample",
            border_color=CURTAILMENT_BORDER,
        )
    with row4[3]:
        kpi_card(
            "Hours with CMG < 100",
            f"{format_int(kpis['cmg_lt_100'])} h",
            f"{format_number(kpis['cmg_lt_100_pct'])}% of sample",
            border_color=CURTAILMENT_BORDER,
        )


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

    st.markdown('<div class="field-label">Select a node</div>', unsafe_allow_html=True)
    if barra_options:
        selected_barra = st.selectbox("", options=barra_options, label_visibility="collapsed")
    else:
        selected_barra = st.text_input("", value="", placeholder="Type a node name", label_visibility="collapsed")
        if not selected_barra:
            st.info("No se pudieron precargar las barras. Escribe el nombre exacto de la barra.")

    st.markdown('<div class="field-label">Select date range</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="field-note">(Available data from {MIN_AVAILABLE_DATE.strftime("%d-%m-%Y")} to today.)</div>',
        unsafe_allow_html=True,
    )

    selected_range = st.date_input(
        "",
        value=(date(2024, 1, 1), date(2024, 12, 31)),
        min_value=MIN_AVAILABLE_DATE,
        max_value=date.today(),
        label_visibility="collapsed",
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

            day_diff = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days

            if day_diff == 0:
                st.session_state["all_profiles"] = build_single_day_curve(work, start_date)
            elif day_diff > 30:
                st.session_state["all_profiles"] = build_monthly_representative_profiles(
                    work,
                    start_date,
                    end_date,
                    "Monthly representative profiles"
                )
            else:
                st.session_state["all_profiles"] = build_single_day_curve(work, start_date)

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