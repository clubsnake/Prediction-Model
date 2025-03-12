# dashboard_integration.py
"""
Integration module to connect the LiveDataManager with the Streamlit dashboard.
"""

try:
    from .dashboard_error import handle_error
except ImportError:
    def handle_error(e):
        # Fallback error handling
        print("Error:", e)

import os
import sys
import threading
import time
from datetime import datetime, timedelta
import streamlit as st

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Removed top-level import from src.dashboard.dashboard.dashboard_ui to avoid circular dependencies

class DashboardIntegrator:
    """
    Connects LiveDataManager with the Streamlit dashboard.
    
    Usage:
      Import in dashboard_core.py as:
         from .dashboard_integration import DashboardIntegrator

    This component links live data updates (via LiveDataManager) with the dashboard UI.
    It works in tandem with study_manager.py to coordinate study data updates.
    """
    
    def __init__(self):
        """Initialize the integrator with robust error handling."""
        try:
            from ...data.data_manager import LiveDataManager
        except Exception as e:
            handle_error(e)
            self.live_data_manager = None
            return
        try:
            if "live_data_manager" not in st.session_state:
                st.session_state["live_data_manager"] = LiveDataManager()
            self.live_data_manager = st.session_state["live_data_manager"]
            st.session_state.setdefault("live_manager_running", False)
            st.session_state.setdefault("last_data_updates", {})
        except Exception as e:
            handle_error(e)

    def ensure_manager_running(self):
        """Ensure the LiveDataManager is running with error handling."""
        try:
            if not st.session_state.get("live_manager_running", False):
                if self.live_data_manager:
                    self.live_data_manager.start()
                    st.session_state["live_manager_running"] = True
                    try:
                        import atexit
                        atexit.register(self.stop_manager)
                    except Exception as inner_e:
                        handle_error(inner_e)
                else:
                    raise Exception("LiveDataManager is not available.")
        except Exception as e:
            handle_error(e)

    def stop_manager(self):
        """Stop the LiveDataManager with robust error handling to ensure proper shutdown."""
        try:
            if st.session_state.get("live_manager_running", False) and self.live_data_manager:
                self.live_data_manager.stop()
                st.session_state["live_manager_running"] = False
        except Exception as e:
            handle_error(e)

    def add_live_data_controls(self):
        """Add LiveDataManager controls to the dashboard with comprehensive error handling."""
        try:
            with st.sidebar.expander("🔴 Live Data Controls", expanded=False):
                status = "Running" if st.session_state.get("live_manager_running", False) else "Stopped"
                status_color = "green" if st.session_state.get("live_manager_running", False) else "red"
                st.markdown(f"**Status:** <span style='color:{status_color};'>{status}</span>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if not st.session_state.get("live_manager_running", False):
                        if st.button("▶️ Start Live Updates", use_container_width=True):
                            self.ensure_manager_running()
                            st.experimental_rerun()
                with col2:
                    if st.session_state.get("live_manager_running", False):
                        if st.button("⏹️ Stop Live Updates", use_container_width=True):
                            self.stop_manager()
                            st.experimental_rerun()
                if st.session_state.get("live_manager_running", False):
                    st.markdown("### Last Data Updates")
                    ticker = st.session_state.get("selected_ticker")
                    timeframe = st.session_state.get("selected_timeframe")
                    if ticker and timeframe:
                        last_update = self.live_data_manager.get_last_update_time(ticker, timeframe) if self.live_data_manager else None
                        if last_update:
                            time_since = datetime.now() - last_update
                            time_str = self._format_timedelta(time_since)
                            st.markdown(f"**{ticker} ({timeframe}):** {time_str} ago")
                            next_update = self._estimate_next_update(timeframe, last_update)
                            if next_update:
                                time_to_next = next_update - datetime.now()
                                if time_to_next.total_seconds() > 0:
                                    st.markdown(f"**Next update in:** {self._format_timedelta(time_to_next)}")
                        else:
                            st.markdown(f"**{ticker} ({timeframe}):** No updates yet")
                        if st.button("🔄 Update Now", use_container_width=True):
                            with st.spinner(f"Updating {ticker} ({timeframe}) data..."):
                                success = self.live_data_manager.manual_update(ticker, timeframe) if self.live_data_manager else False
                                if success:
                                    st.success("Data updated successfully")
                                else:
                                    st.error("Failed to update data")
                st.markdown("### Settings")
                auto_update = st.toggle(
                    "Auto-refresh dashboard on data update",
                    value=st.session_state.get("auto_update_on_data_change", True)
                )
                st.session_state["auto_update_on_data_change"] = auto_update
        except Exception as e:
            handle_error(e)

    def add_data_freshness_indicator(self):
        """Add an indicator showing data freshness with error handling."""
        try:
            ticker = st.session_state.get("selected_ticker")
            timeframe = st.session_state.get("selected_timeframe")
            if ticker and timeframe and st.session_state.get("live_manager_running", False):
                last_update = self.live_data_manager.get_last_update_time(ticker, timeframe) if self.live_data_manager else None
                if last_update:
                    time_since = datetime.now() - last_update
                    time_str = self._format_timedelta(time_since)
                    if timeframe == "1m" and time_since > timedelta(minutes=2):
                        status_color = "red"
                    elif timeframe in ["5m", "15m", "30m"] and time_since > timedelta(minutes=int(timeframe[:-1]) * 2):
                        status_color = "red"
                    elif timeframe == "1h" and time_since > timedelta(hours=2):
                        status_color = "red"
                    elif timeframe == "1d" and time_since > timedelta(days=2):
                        status_color = "red"
                    elif time_since > timedelta(hours=1):
                        status_color = "orange"
                    else:
                        status_color = "green"
                    st.markdown(
                        f"""
                        <div style="position:absolute;top:10px;right:10px;padding:5px 10px;
                        background-color:rgba(0,0,0,0.1);border-radius:5px;font-size:0.8em;">
                        <span style="color:{status_color};">●</span> Data updated {time_str} ago
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        except Exception as e:
            handle_error(e)

    def update_on_data_change(self):
        """Check for data updates and refresh dashboard safely."""
        try:
            if not st.session_state.get("auto_update_on_data_change", True):
                return
            ticker = st.session_state.get("selected_ticker")
            timeframe = st.session_state.get("selected_timeframe")
            if ticker and timeframe and st.session_state.get("live_manager_running", False):
                last_update = self.live_data_manager.get_last_update_time(ticker, timeframe) if self.live_data_manager else None
                if last_update:
                    last_check_key = f"{ticker}_{timeframe}_last_check"
                    last_check = st.session_state.get(last_check_key, datetime.now() - timedelta(days=1))
                    if last_update > last_check:
                        st.session_state[last_check_key] = datetime.now()
                        df = self.live_data_manager.get_latest_data(ticker, timeframe) if self.live_data_manager else None
                        if df is not None:
                            st.session_state[f"{ticker}_{timeframe}_data"] = df
                            if ticker == st.session_state.get("selected_ticker") and timeframe == st.session_state.get("selected_timeframe"):
                                st.session_state["current_data"] = df
                            st.experimental_rerun()
        except Exception as e:
            handle_error(e)

    def _format_timedelta(self, td):
        """Format a timedelta into a human-readable string with error handling."""
        try:
            total_seconds = int(td.total_seconds())
            if total_seconds < 60:
                return f"{total_seconds}s"
            elif total_seconds < 3600:
                return f"{total_seconds // 60}m {total_seconds % 60}s"
            elif total_seconds < 86400:
                return f"{total_seconds // 3600}h {(total_seconds % 3600) // 60}m"
            else:
                return f"{total_seconds // 86400}d {(total_seconds % 86400) // 3600}h"
        except Exception as e:
            handle_error(e)
            return "N/A"

    def _estimate_next_update(self, timeframe, last_update):
        """Estimate the next scheduled update time with error handling."""
        try:
            now = datetime.now()
            if timeframe == "1m":
                return last_update + timedelta(minutes=1)
            elif timeframe.endswith("m"):
                minutes = int(timeframe[:-1])
                current_minute = now.minute
                next_slot = ((current_minute // minutes) + 1) * minutes
                if next_slot >= 60:
                    return now.replace(hour=now.hour + 1, minute=next_slot - 60, second=0, microsecond=0)
                else:
                    return now.replace(minute=next_slot, second=0, microsecond=0)
            elif timeframe == "1h":
                return now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            elif timeframe.endswith("h"):
                hours = int(timeframe[:-1])
                current_hour = now.hour
                next_slot = ((current_hour // hours) + 1) * hours
                if next_slot >= 24:
                    return now.replace(day=now.day + 1, hour=next_slot - 24, minute=0, second=0, microsecond=0)
                else:
                    return now.replace(hour=next_slot, minute=0, second=0, microsecond=0)
            elif timeframe == "1d":
                return (now.replace(hour=0, minute=1, second=0, microsecond=0) + timedelta(days=1))
            elif timeframe == "1wk":
                days_to_monday = 7 - now.weekday()
                if days_to_monday == 7:
                    days_to_monday = 0
                return (now.replace(hour=0, minute=1, second=0, microsecond=0) + timedelta(days=days_to_monday))
            elif timeframe == "1mo":
                if now.month == 12:
                    return datetime(now.year + 1, 1, 1, 0, 1, 0)
                else:
                    return datetime(now.year, now.month + 1, 1, 0, 1, 0)
            return (now.replace(hour=0, minute=1, second=0, microsecond=0) + timedelta(days=1))
        except Exception as e:
            handle_error(e)
            return None

# Helper function to integrate with existing dashboard
def integrate_live_data_with_dashboard():
    """
    Integrate LiveDataManager with the dashboard.
    
    Import in dashboard_core.py as:
         from .dashboard_integration import integrate_live_data_with_dashboard
         
    This function connects the live data updates through DashboardIntegrator with the dashboard UI.
    It works alongside study_manager.py to coordinate study data updates.
    """
    try:
        if "dashboard_integrator" not in st.session_state:
            st.session_state["dashboard_integrator"] = DashboardIntegrator()
        integrator = st.session_state["dashboard_integrator"]
        integrator.ensure_manager_running()
        integrator.add_live_data_controls()
        integrator.add_data_freshness_indicator()
        integrator.update_on_data_change()
        return integrator
    except Exception as e:
        handle_error(e)
        return None