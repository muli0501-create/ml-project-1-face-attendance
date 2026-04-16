"""
签到记录可视化模块
==================
分析 attendance_log/*.csv 并生成可视化图表
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime


def get_project_root():
    return Path(__file__).parent.parent.absolute()


def load_attendance_logs(log_dir: Path = None) -> pd.DataFrame:
    """加载所有签到记录 CSV"""
    if log_dir is None:
        log_dir = get_project_root() / "attendance_log"

    all_records = []
    if not log_dir.exists():
        return pd.DataFrame()

    for csv_file in log_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, encoding="utf-8-sig")
            df["来源文件"] = csv_file.name
            all_records.append(df)
        except Exception as e:
            print(f"读取 {csv_file} 失败: {e}")

    if not all_records:
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    # 解析签到时间
    combined["签到时间_dt"] = pd.to_datetime(combined["签到时间"], errors="coerce")
    combined["日期"] = combined["签到时间_dt"].dt.date
    combined["小时"] = combined["签到时间_dt"].dt.hour
    combined["星期"] = combined["签到时间_dt"].dt.day_name()
    return combined


def plot_attendance_stats(df: pd.DataFrame) -> go.Figure:
    """生成签到统计图表"""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="暂无签到记录", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["到场人数统计", "签到时间分布(小时)", "每日到场趋势", "各时段签到分布"],
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )

    # 1. 到场人数柱状图
    if "姓名" in df.columns:
        name_counts = df["姓名"].value_counts().reset_index()
        name_counts.columns = ["姓名", "到场次数"]
        fig.add_trace(
            go.Bar(x=name_counts["姓名"], y=name_counts["到场次数"], marker_color="steelblue"),
            row=1, col=1
        )

    # 2. 签到时间分布直方图
    if "小时" in df.columns and df["小时"].notna().any():
        fig.add_trace(
            go.Histogram(x=df["小时"].dropna(), nbinsx=24, marker_color="coral", name="签到时间"),
            row=1, col=2
        )
        fig.update_xaxes(title_text="小时 (0-23)", row=1, col=2)
        fig.update_yaxes(title_text="人数", row=1, col=2)

    # 3. 每日到场趋势
    if "日期" in df.columns:
        daily_counts = df.groupby("日期").size().reset_index(name="到场人数")
        fig.add_trace(
            go.Scatter(x=daily_counts["日期"], y=daily_counts["到场人数"],
                       mode="lines+markers", marker_color="green", name="每日到场"),
            row=2, col=1
        )
        fig.update_xaxes(title_text="日期", row=2, col=1)
        fig.update_yaxes(title_text="人数", row=2, col=1)

    # 4. 到场率饼图（如果有多天数据）
    if "姓名" in df.columns:
        person_attendance = df.groupby("姓名").size().reset_index(name="签到次数")
        fig.add_trace(
            go.Pie(labels=person_attendance["姓名"], values=person_attendance["签到次数"],
                   textinfo="label+percent"),
            row=2, col=2
        )

    fig.update_layout(height=700, width=1100, showlegend=False,
                      title_text="签到记录统计分析")
    return fig


def plot_time_distribution(df: pd.DataFrame) -> go.Figure:
    """生成签到时间分布详细图表"""
    if df.empty or "小时" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="暂无数据", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df["小时"].dropna(), nbinsx=24,
                               marker_color="mediumpurple", name="签到人数"))

    fig.update_layout(
        title="签到时间分布（按小时）",
        xaxis_title="小时",
        yaxis_title="签到人数",
        height=400
    )
    return fig


if __name__ == "__main__":
    print("=" * 50)
    print("签到记录可视化")
    print("=" * 50)

    root = get_project_root()
    log_dir = root / "attendance_log"

    if not log_dir.exists():
        print(f"签到记录目录不存在: {log_dir}")
    else:
        df = load_attendance_logs(log_dir)
        if df.empty:
            print("没有找到签到记录")
        else:
            print(f"共加载 {len(df)} 条签到记录")
            print(df.head())
