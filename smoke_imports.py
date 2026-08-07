import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from bot.calibration import calibrate_from_leaderboard, center_lock_calibrate
from bot.planner import ClickPlanner, ClickPlannerConfig
from bot.economy import TroopTracker
from bot.controls import MouseControls
from bot.click_loop import ClickLoop
print("all imports OK")
