"""
Nudge Detection — Python replication package.

Mirrors the C++ nudge detection pipeline for rapid prototyping:

  C++ File                          Python Module
  --------------------------------  -----------------------------------------
  DgpsTrajectoryAnalyzer.cpp/.hpp   dgps_trajectory_analyzer.py
  NudgeObjectAnalyzer.cpp/.hpp      nudge_object_analyzer.py
  BiasNudgeDecider.cpp/.hpp         bias_nudge_decider.py
  NudgeClassifier.cpp/.hpp          nudge_classifier.py
  (various .hpp headers)            data_types.py

Usage:
    from python.data_types import Config, DetectionResult
    from python.dgps_trajectory_analyzer import DgpsTrajectoryAnalyzer
    from python.nudge_object_analyzer import NudgeObjectAnalyzer
    from python.bias_nudge_decider import BiasNudgeDecider
    from python.nudge_classifier import NudgeClassifier
"""
