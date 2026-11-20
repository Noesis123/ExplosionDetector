"""
Generate a publication-grade, presentation-ready architectural infographic poster
matching the exact design, layout, color scheme, and typography of the reference
acoustic detector diagram - with zero text overlap and pixel-perfect alignment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Polygon
import numpy as np
from pathlib import Path
import shutil

# Setup canvas size (16:10 aspect ratio, high resolution)
fig = plt.figure(figsize=(16, 10.2), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 102)
ax.axis('off')

# Background
bg = FancyBboxPatch((0.5, 0.5), 159, 101, boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor='#F4F6F9', edgecolor='#E2E8F0', linewidth=1.5)
ax.add_patch(bg)

# ==========================================
# HEADER BANNER
# ==========================================
header_patch = FancyBboxPatch((32, 91.5), 96, 8.2, boxstyle="round,pad=0,rounding_size=1.2",
                              facecolor='#0E254A', edgecolor='#1E3A8A', linewidth=1.5)
ax.add_patch(header_patch)

ax.text(80, 96.6, "Acoustic Explosion Detector", ha='center', va='center',
        fontsize=21, fontweight='bold', color='#FFFFFF', fontfamily='sans-serif')
ax.text(80, 93.3, "System Architecture & Workflow", ha='center', va='center',
        fontsize=12, fontweight='medium', color='#93C5FD', fontfamily='sans-serif')

# ==========================================
# CARD SPECIFICATIONS
# ==========================================
card_w = 37.0
card_h = 36.8
xs = [2.5, 41.8, 81.1, 120.4]
y_row1 = 52.8
y_row2 = 14.2

def draw_card_base(x, y, w, h, border_color, num_str, num_bg, title_str, title_color):
    card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                          facecolor='#FFFFFF', edgecolor=border_color, linewidth=2.0)
    ax.add_patch(card)
    
    badge = Circle((x + 2.8, y + h - 3.2), 1.8, facecolor=num_bg, edgecolor='none')
    ax.add_patch(badge)
    ax.text(x + 2.8, y + h - 3.2, num_str, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#FFFFFF', fontfamily='sans-serif')
    
    ax.text(x + 5.5, y + h - 3.2, title_str, ha='left', va='center',
            fontsize=12.5, fontweight='bold', color=title_color, fontfamily='sans-serif')

# ==========================================
# CARD 1: AUDIO CAPTURE
# ==========================================
draw_card_base(xs[0], y_row1, card_w, card_h, '#93C5FD', '1', '#2563EB', 'Audio Capture', '#1D4ED8')

# Microphone icon with sound waves
mic_cx, mic_cy = xs[0] + card_w/2, y_row1 + 24.8
ax.add_patch(FancyBboxPatch((mic_cx - 1.8, mic_cy - 1.5), 3.6, 5.5, boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor='#1D4ED8', edgecolor='none'))
theta = np.linspace(-np.pi, 0, 100)
arc_x = mic_cx + 3.0 * np.cos(theta)
arc_y = mic_cy + 1.2 + 3.0 * np.sin(theta)
ax.plot(arc_x, arc_y, color='#1E40AF', lw=2.2, solid_capstyle='round')
ax.plot([mic_cx, mic_cx], [mic_cy - 1.8, mic_cy - 4.5], color='#1E40AF', lw=2.2)
ax.plot([mic_cx - 2.8, mic_cx + 2.8], [mic_cy - 4.5, mic_cy - 4.5], color='#1E40AF', lw=2.2, solid_capstyle='round')

for r_scale, alpha_val in [(4.8, 0.9), (6.8, 0.5)]:
    th_l = np.linspace(3*np.pi/4, 5*np.pi/4, 40)
    ax.plot(mic_cx - 1.0 + r_scale * np.cos(th_l), mic_cy + 1.5 + r_scale * np.sin(th_l),
            color='#2563EB', lw=2.0, alpha=alpha_val, solid_capstyle='round')
    th_r = np.linspace(-np.pi/4, np.pi/4, 40)
    ax.plot(mic_cx + 1.0 + r_scale * np.cos(th_r), mic_cy + 1.5 + r_scale * np.sin(th_r),
            color='#2563EB', lw=2.0, alpha=alpha_val, solid_capstyle='round')

ax.text(xs[0] + card_w/2, y_row1 + 16.5, "Microphone continuously\ncaptures ambient sound.",
        ha='center', va='center', fontsize=10.5, color='#475569', fontfamily='sans-serif', linespacing=1.3)

box1 = FancyBboxPatch((xs[0] + 2.2, y_row1 + 2.4), card_w - 4.4, 10.5, boxstyle="round,pad=0,rounding_size=0.6",
                      facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1.2)
ax.add_patch(box1)

labels_c1 = [("Sampling Rate", ":  22,050 Hz"),
             ("Channels", ":  Mono (Float32)"),
             ("Ring Buffer", ":  5.0 s (Rolling FIFO)")]
for i, (k, v) in enumerate(labels_c1):
    curr_y = y_row1 + 10.2 - i * 3.1
    ax.text(xs[0] + 4.0, curr_y, k, ha='left', va='center', fontsize=9.5, color='#64748B', fontweight='semibold')
    ax.text(xs[0] + 17.0, curr_y, v, ha='left', va='center', fontsize=9.5, color='#0F172A', fontweight='bold', fontfamily='monospace')

# ==========================================
# CARD 2: NOISE CANCELLATION
# ==========================================
draw_card_base(xs[1], y_row1, card_w, card_h, '#86EFAC', '2', '#16A34A', 'Noise Cancellation', '#15803D')

box2_top = FancyBboxPatch((xs[1] + 2.0, y_row1 + 19.8), card_w - 4.0, 11.2, boxstyle="round,pad=0,rounding_size=0.6",
                          facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1.2)
ax.add_patch(box2_top)

ax.text(xs[1] + 9.0, y_row1 + 29.0, "Noisy audio", ha='center', va='center',
        fontsize=9, fontweight='bold', color='#DC2626')
ax.text(xs[1] + card_w - 9.0, y_row1 + 29.0, "Cleaned audio", ha='center', va='center',
        fontsize=9, fontweight='bold', color='#16A34A')

t_w = np.linspace(0, 1, 120)
noisy_sig = 0.8 * np.sin(2 * np.pi * 3.5 * t_w) + 0.4 * np.sin(2 * np.pi * 14 * t_w) + 0.25 * np.random.RandomState(42).randn(120)
clean_sig = 0.9 * np.sin(2 * np.pi * 3.5 * t_w)

wx_left = np.linspace(xs[1] + 3.5, xs[1] + 15.0, 120)
wy_noisy = y_row1 + 24.0 + noisy_sig * 1.5
ax.plot(wx_left, wy_noisy, color='#DC2626', lw=1.6)

ax.annotate('', xy=(xs[1] + 20.8, y_row1 + 24.0), xytext=(xs[1] + 16.5, y_row1 + 24.0),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=2.2))

wx_right = np.linspace(xs[1] + 22.2, xs[1] + 33.7, 120)
wy_clean = y_row1 + 24.0 + clean_sig * 1.5
ax.plot(wx_right, wy_clean, color='#16A34A', lw=2.0)

ax.text(xs[1] + card_w/2, y_row1 + 15.4,
        "Adaptive 90 Hz high-pass conditioning eliminates\nDC bias & cooling fan line rumble (45–88 Hz).",
        ha='center', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

box2_bot = FancyBboxPatch((xs[1] + 2.2, y_row1 + 2.4), card_w - 4.4, 9.8, boxstyle="round,pad=0,rounding_size=0.6",
                          facecolor='#F8FAFC', edgecolor='#E2E8F0', linewidth=1.2)
ax.add_patch(box2_bot)

specs_c2 = [("Filter Topology", ":  2nd-Order Butterworth"),
            ("Cutoff Frequency", ":  90.0 Hz (Audible pass)"),
            ("Filter State", ":  Stateful zi (No clicks)")]
for i, (k, v) in enumerate(specs_c2):
    curr_y = y_row1 + 9.6 - i * 3.0
    ax.text(xs[1] + 3.8, curr_y, k, ha='left', va='center', fontsize=9.0, color='#64748B', fontweight='semibold')
    ax.text(xs[1] + 17.5, curr_y, v, ha='left', va='center', fontsize=8.8, color='#0F172A', fontweight='bold', fontfamily='monospace')

# ==========================================
# CARD 3: STAGE 1 PHYSICAL CHECKS
# ==========================================
draw_card_base(xs[2], y_row1, card_w, card_h, '#D8B4FE', '3', '#7C3AED', 'Stage 1: Physical Checks', '#6D28D9')

ax.text(xs[2] + 2.5, y_row1 + card_h - 6.5,
        "Checks 5 physical characteristics. If any check fails,\nthe sound is rejected (zero CNN overhead).",
        ha='left', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

checks = [
    ("RMS Energy Level", "≥ -45.0 dBFS"),
    ("Window Duration", "1.0 s (22,050 spl)"),
    ("Transient Attack", "Trise ≤ 120 ms"),
    ("Evaluation Cadence", "4 Hz (0.25 s hop)"),
    ("Audible Band Ratio", "Freqs ≥ 20 Hz")
]

for i, (param, cond) in enumerate(checks):
    r_y = y_row1 + 21.0 - i * 4.4
    p_box = FancyBboxPatch((xs[2] + 2.2, r_y - 1.6), card_w - 4.4, 3.6, boxstyle="round,pad=0,rounding_size=0.5",
                           facecolor='#FAF5FF', edgecolor='#F3E8FF', linewidth=1.0)
    ax.add_patch(p_box)
    
    ax.add_patch(Circle((xs[2] + 4.0, r_y + 0.2), 0.55, facecolor='#7C3AED', edgecolor='none'))
    ax.text(xs[2] + 5.5, r_y + 0.2, param, ha='left', va='center', fontsize=9.4, color='#374151', fontweight='semibold')
    
    cond_badge = FancyBboxPatch((xs[2] + card_w - 15.5, r_y - 1.1), 13.0, 2.6, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor='#EDE9FE', edgecolor='#DDD6FE', linewidth=0.8)
    ax.add_patch(cond_badge)
    ax.text(xs[2] + card_w - 9.0, r_y + 0.2, cond, ha='center', va='center',
            fontsize=8.5, color='#6D28D9', fontweight='bold', fontfamily='monospace')

# ==========================================
# CARD 4: STAGE 2 DEEP LEARNING (2D-CNN)
# ==========================================
draw_card_base(xs[3], y_row1, card_w, card_h, '#FDE68A', '4', '#D97706', 'Stage 2: Deep Learning (2D-CNN)', '#B45309')

ax.text(xs[3] + 2.5, y_row1 + card_h - 6.5,
        "Analyzes time-frequency patterns using a 4-block\n2D CNN with batch normalization and dropout.",
        ha='left', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

# 1. Create Spectrogram
ax.text(xs[3] + 5.2, y_row1 + 23.2, "1. Spectrogram", ha='center', va='center',
        fontsize=8.2, fontweight='bold', color='#92400E')

spec_x, spec_y, spec_w, spec_h = xs[3] + 1.2, y_row1 + 10.0, 8.2, 11.5
ax.add_patch(Rectangle((spec_x, spec_y), spec_w, spec_h, facecolor='#0F172A', edgecolor='#334155', lw=1))
np.random.seed(101)
heat = np.zeros((24, 20))
for r in range(24):
    for c in range(20):
        dist = np.sqrt(((r - 5)**2)*1.8 + (c - 8)**2)
        heat[r, c] = np.exp(-dist/4.5) + 0.15 * np.random.rand()
heat = np.clip(heat, 0, 1)
ax.imshow(heat, extent=[spec_x, spec_x + spec_w, spec_y, spec_y + spec_h],
          origin='lower', cmap='inferno', aspect='auto')

ax.text(xs[3] + 5.2, y_row1 + 8.0, "64 Mel x 44", ha='center', va='center',
        fontsize=7.2, color='#64748B', fontfamily='monospace')

# 2. 2D CNN (Isometric 4 Blocks)
ax.text(xs[3] + 15.5, y_row1 + 23.2, "2. 2D CNN\n(4 Blocks)", ha='center', va='center',
        fontsize=7.8, fontweight='bold', color='#92400E', linespacing=1.1)

block_colors = [('#3B82F6', '#1D4ED8'), ('#22C55E', '#15803D'), ('#EAB308', '#A16207'), ('#F97316', '#C2410C')]
for b_i, (c_top, c_side) in enumerate(block_colors):
    bx = xs[3] + 10.5 + b_i * 2.6
    by = y_row1 + 10.8 + b_i * 1.5
    bw = 2.0
    bh = 7.0 + b_i * 0.8
    ax.add_patch(Rectangle((bx, by), bw, bh, facecolor=c_top, edgecolor='none'))
    pts_side = [[bx + bw, by], [bx + bw + 0.8, by + 0.8], [bx + bw + 0.8, by + bh + 0.8], [bx + bw, by + bh]]
    ax.add_patch(Polygon(pts_side, facecolor=c_side, edgecolor='none'))
    pts_top = [[bx, by + bh], [bx + 0.8, by + bh + 0.8], [bx + bw + 0.8, by + bh + 0.8], [bx + bw, by + bh]]
    ax.add_patch(Polygon(pts_top, facecolor='#93C5FD' if b_i==0 else '#86EFAC' if b_i==1 else '#FEF08A' if b_i==2 else '#FDBA74', edgecolor='none'))

ax.text(xs[3] + 15.5, y_row1 + 8.0, "130k Params", ha='center', va='center',
        fontsize=7.2, color='#64748B', fontfamily='monospace')

# 3. Predict Class Probabilities
ax.text(xs[3] + 28.5, y_row1 + 23.2, "3. Predictions", ha='center', va='center',
        fontsize=8.2, fontweight='bold', color='#92400E')

classes_pred = [
    ("Explosion", 0.86, '#DC2626', True),
    ("Thunder", 0.06, '#64748B', False),
    ("Backfire", 0.04, '#64748B', False),
    ("Gunfire", 0.02, '#64748B', False),
    ("Impacts", 0.01, '#64748B', False),
    ("Speech", 0.01, '#64748B', False)
]

for p_i, (cname, pval, ccolor, is_bold) in enumerate(classes_pred):
    py = y_row1 + 19.8 - p_i * 2.3
    ax.text(xs[3] + 22.0, py, cname, ha='left', va='center',
            fontsize=7.2, fontweight='bold' if is_bold else 'normal', color=ccolor)
    ax.add_patch(Rectangle((xs[3] + 28.0, py - 0.4), 4.8, 0.8, facecolor='#E2E8F0', edgecolor='none'))
    ax.add_patch(Rectangle((xs[3] + 28.0, py - 0.4), 4.8 * pval, 0.8, facecolor=ccolor, edgecolor='none'))
    ax.text(xs[3] + 35.8, py, f"{pval:.2f}", ha='right', va='center',
            fontsize=7.2, fontweight='bold' if is_bold else 'normal', color=ccolor, fontfamily='monospace')

# ==========================================
# CARD 5: DECISION LOGIC
# ==========================================
draw_card_base(xs[0], y_row2, card_w, card_h, '#FECACA', '5', '#DC2626', 'Decision Logic (AEV)', '#B91C1C')

ax.text(xs[0] + 2.5, y_row2 + card_h - 6.5,
        "Applies strict acoustic thresholds (AEV Config C)\nto confirm genuine blast wave characteristics.",
        ha='left', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

th_box = FancyBboxPatch((xs[0] + 1.8, y_row2 + 19.8), card_w - 3.6, 3.2, boxstyle="round,pad=0,rounding_size=0.4",
                        facecolor='#FEE2E2', edgecolor='#FECACA', linewidth=1.0)
ax.add_patch(th_box)
ax.text(xs[0] + 2.8, y_row2 + 21.4, "Decision Rule", ha='left', va='center', fontsize=8.6, fontweight='bold', color='#991B1B')
ax.text(xs[0] + 19.5, y_row2 + 21.4, "Condition", ha='left', va='center', fontsize=8.6, fontweight='bold', color='#991B1B')
ax.text(xs[0] + 27.2, y_row2 + 21.4, "Meaning", ha='left', va='center', fontsize=8.6, fontweight='bold', color='#991B1B')

dec_rules = [
    ("Blast confidence", "≥ 0.70", "AI certainty required"),
    ("Low-frequency ratio", "≥ 0.35", "Rejects speech & claps"),
    ("High-frequency ratio", "≤ 0.15", "Rejects glass & hiss"),
    ("Energy squelch floor", "≥ -45 dBFS", "Rejects sensor noise"),
    ("Target logit margin", "Δ ≥ +0.84", "Dominates ambient")
]

for d_i, (rule, cond, meaning) in enumerate(dec_rules):
    d_y = y_row2 + 16.5 - d_i * 3.6
    bg_col = '#FFF5F5' if d_i % 2 == 1 else '#FFFFFF'
    row_patch = Rectangle((xs[0] + 1.8, d_y - 1.5), card_w - 3.6, 3.4, facecolor=bg_col, edgecolor='#FEE2E2', lw=0.6)
    ax.add_patch(row_patch)
    
    ax.text(xs[0] + 2.8, d_y + 0.2, rule, ha='left', va='center', fontsize=8.0, color='#374151', fontweight='semibold')
    ax.text(xs[0] + 19.5, d_y + 0.2, cond, ha='left', va='center', fontsize=8.0, color='#B91C1C', fontweight='bold', fontfamily='monospace')
    ax.text(xs[0] + 27.2, d_y + 0.2, meaning, ha='left', va='center', fontsize=7.6, color='#64748B')

# ==========================================
# CARD 6: TEMPORAL CONFIRMATION
# ==========================================
draw_card_base(xs[1], y_row2, card_w, card_h, '#A5F3FC', '6', '#0891B2', 'Temporal Confirmation', '#0E7490')

ax.text(xs[1] + 2.5, y_row2 + card_h - 6.5,
        "Tracks detonation shockwaves, suppresses echoes,\nand handles multi-blast event incidents.",
        ha='left', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

temp_steps = [
    ('1', "1. Instant Initial Alert", "Fires immediately on 1st qualifying\nwindow (eval latency ≤ 0.25 s)."),
    ('2', "2. Echo & Reverb Suppression", "Consecutive detections within 6.0 s\nmerged under same Event ID."),
    ('3', "3. Secondary Detonation Surge", "Energy surge > 6.0 dB triggers\na distinct secondary blast alert.")
]

for t_i, (num, stitle, sdesc) in enumerate(temp_steps):
    t_y = y_row2 + 19.2 - t_i * 6.2
    tbox = FancyBboxPatch((xs[1] + 2.2, t_y - 3.2), card_w - 4.4, 5.6, boxstyle="round,pad=0,rounding_size=0.5",
                          facecolor='#F0FDFA', edgecolor='#CCFBF1', linewidth=1.0)
    ax.add_patch(tbox)
    
    ax.add_patch(Circle((xs[1] + 4.2, t_y + 1.2), 1.1, facecolor='#0D9488', edgecolor='none'))
    ax.text(xs[1] + 4.2, t_y + 1.2, num, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF')
    
    ax.text(xs[1] + 6.2, t_y + 1.2, stitle, ha='left', va='center', fontsize=9.0, fontweight='bold', color='#0F766E')
    ax.text(xs[1] + 3.2, t_y - 1.2, sdesc, ha='left', va='center', fontsize=7.6, color='#334155', linespacing=1.15)

ax.text(xs[1] + card_w/2, y_row2 + 1.8,
        "After blast ends, requires 3 quiet windows (750 ms) to rearm.",
        ha='center', va='center', fontsize=7.6, color='#0E7490', fontweight='semibold')

# ==========================================
# CARD 7: ALERT & OPERATOR
# ==========================================
draw_card_base(xs[2], y_row2, card_w, card_h, '#C4B5FD', '7', '#6366F1', 'Alert & Operator Hub', '#4338CA')

mon_x, mon_y = xs[2] + 11.5, y_row2 + 26.2
ax.add_patch(FancyBboxPatch((mon_x - 4.5, mon_y - 3.5), 9.0, 6.5, boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor='#1E1B4B', edgecolor='#4F46E5', linewidth=1.5))
tri_pts = [[mon_x, mon_y + 1.8], [mon_x - 2.2, mon_y - 2.0], [mon_x + 2.2, mon_y - 2.0]]
ax.add_patch(Polygon(tri_pts, facecolor='#EF4444', edgecolor='none'))
ax.text(mon_x, mon_y - 0.6, "!", ha='center', va='center', fontsize=11, fontweight='bold', color='#FFFFFF')
ax.plot([mon_x, mon_x], [mon_y - 3.5, mon_y - 5.0], color='#4F46E5', lw=2.0)
ax.plot([mon_x - 2.0, mon_x + 2.0], [mon_y - 5.0, mon_y - 5.0], color='#4F46E5', lw=2.0)

op_x, op_y = xs[2] + 25.5, y_row2 + 26.2
ax.add_patch(Circle((op_x, op_y - 0.5), 2.8, facecolor='#E0E7FF', edgecolor='none'))
theta_hp = np.linspace(0, np.pi, 50)
ax.plot(op_x + 3.4 * np.cos(theta_hp), op_y - 0.5 + 3.4 * np.sin(theta_hp), color='#4F46E5', lw=2.4)
ax.add_patch(FancyBboxPatch((op_x - 4.1, op_y - 2.0), 1.4, 3.0, boxstyle="round,pad=0,rounding_size=0.4", facecolor='#4338CA'))
ax.add_patch(FancyBboxPatch((op_x + 2.7, op_y - 2.0), 1.4, 3.0, boxstyle="round,pad=0,rounding_size=0.4", facecolor='#4338CA'))

ax.text(xs[2] + card_w/2, y_row2 + 18.8,
        "If all conditions are met, the system sends an\nadvisory alert to the security operations center.",
        ha='center', va='center', fontsize=9.8, color='#475569', fontfamily='sans-serif', linespacing=1.3)

alert_items = [
    ("Live SSE Stream", "/api/stream broadcast (4 Hz)"),
    ("Immutable SQLite", "Audit in alerts_history.db"),
    ("Threat Telemetry", "Peak RMS, P_exp, LF/HF"),
    ("Operator Audio/UI", "Audible chime & red banner")
]

for a_i, (label_t, desc_t) in enumerate(alert_items):
    a_y = y_row2 + 13.0 - a_i * 3.4
    ax.text(xs[2] + 3.5, a_y, "✓", ha='center', va='center', fontsize=11, fontweight='bold', color='#4F46E5')
    ax.text(xs[2] + 5.5, a_y, label_t, ha='left', va='center', fontsize=8.6, fontweight='bold', color='#1E293B')
    ax.text(xs[2] + 19.2, a_y, f": {desc_t}", ha='left', va='center', fontsize=7.6, color='#64748B')

# ==========================================
# CARD 8: PRIVACY & SAFETY
# ==========================================
draw_card_base(xs[3], y_row2, card_w, card_h, '#CBD5E1', '8', '#475569', 'Privacy & Safety', '#334155')

sh_cx, sh_cy = xs[3] + card_w/2, y_row2 + 25.8
sh_pts = [
    [sh_cx, sh_cy + 4.5],
    [sh_cx + 4.2, sh_cy + 3.0],
    [sh_cx + 4.2, sh_cy - 0.5],
    [sh_cx, sh_cy - 4.5],
    [sh_cx - 4.2, sh_cy - 0.5],
    [sh_cx - 4.2, sh_cy + 3.0]
]
ax.add_patch(Polygon(sh_pts, facecolor='#F1F5F9', edgecolor='#334155', linewidth=2.0))
ax.add_patch(FancyBboxPatch((sh_cx - 1.5, sh_cy - 2.0), 3.0, 2.4, boxstyle="round,pad=0,rounding_size=0.4",
                            facecolor='#334155', edgecolor='none'))
ax.plot([sh_cx - 1.0, sh_cx - 1.0, sh_cx + 1.0, sh_cx + 1.0],
        [sh_cy + 0.4, sh_cy + 1.6, sh_cy + 1.6, sh_cy + 0.4], color='#334155', lw=1.6)

safety_checks = [
    "Does not identify voices or speakers",
    "Does not detect language or spoken words",
    "Processes audio in memory (temporary FIFO)",
    "Flushes audio buffer every 5.0 seconds",
    "100% on-device edge AI (Zero cloud latency)",
    "Alerts are advisory only (human in the loop)"
]

for s_i, scheck in enumerate(safety_checks):
    s_y = y_row2 + 18.0 - s_i * 3.1
    ax.text(xs[3] + 3.5, s_y, "✓", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#16A34A')
    ax.text(xs[3] + 5.5, s_y, scheck, ha='left', va='center', fontsize=8.8, color='#334155')

# ==========================================
# COMPLETE FLOW AT A GLANCE (HORIZONTAL BAR)
# ==========================================
flow_y = 5.8
flow_h = 6.2
flow_bar = FancyBboxPatch((2.5, flow_y), 154.9, flow_h, boxstyle="round,pad=0,rounding_size=1.0",
                          facecolor='#F0F7FF', edgecolor='#BFDBFE', linewidth=1.5)
ax.add_patch(flow_bar)

ax.text(4.2, flow_y + flow_h/2, "Complete Flow\n(at a Glance)", ha='left', va='center',
        fontsize=9.5, fontweight='bold', color='#1E3A8A', linespacing=1.1)

flow_nodes = [
    ("Capture\nAudio", '#2563EB'),
    ("Noise\nConditioning", '#16A34A'),
    ("Stage 1\nPhysical Checks", '#7C3AED'),
    ("Stage 2\n2D-CNN", '#D97706'),
    ("Decision\nLogic", '#DC2626'),
    ("Temporal\nTracking", '#0891B2'),
    ("Alert to\nOperator", '#4F46E5')
]

node_xs = np.linspace(22.0, 147.0, 7)
for idx, (label, col) in enumerate(flow_nodes):
    nx = node_xs[idx]
    capsule = FancyBboxPatch((nx - 6.8, flow_y + 1.1), 13.6, 4.0, boxstyle="round,pad=0,rounding_size=1.0",
                             facecolor='#FFFFFF', edgecolor=col, linewidth=1.2)
    ax.add_patch(capsule)
    ax.text(nx, flow_y + 3.1, label, ha='center', va='center',
            fontsize=8.2, fontweight='bold', color=col, linespacing=1.1)
    
    if idx < 6:
        next_nx = node_xs[idx + 1]
        ax.annotate('', xy=(next_nx - 7.5, flow_y + 3.1), xytext=(nx + 7.5, flow_y + 3.1),
                    arrowprops=dict(arrowstyle="->", color='#60A5FA', lw=2.0))

# ==========================================
# FOOTER VALUE BADGES
# ==========================================
badges = ["Fewer False Alarms", "Faster Detection (≤ 250 ms)", "Zero Cloud Dependency", "Safer Public Spaces"]
b_xs = np.linspace(35, 125, 4)
for b_i, btext in enumerate(badges):
    bx = b_xs[b_i]
    bpill = FancyBboxPatch((bx - 12.0, 1.2), 24.0, 3.2, boxstyle="round,pad=0,rounding_size=1.0",
                           facecolor='#DBEAFE', edgecolor='#BFDBFE', linewidth=1.0)
    ax.add_patch(bpill)
    ax.text(bx, 2.8, btext, ha='center', va='center',
            fontsize=9.2, fontweight='bold', color='#1E40AF')

# Output paths in brain artifact dir and workspace
out_dir = Path(r"C:\Users\Aman Yazeen\.gemini\antigravity\brain\ad076906-e1fe-4046-9a40-92f5f2339702")
proj_dir = Path(r"D:\NOESIS\Explosion\Explosion\reports")
proj_dir.mkdir(parents=True, exist_ok=True)

png_path = out_dir / "acoustic_explosion_detector_architecture.png"
svg_path = out_dir / "acoustic_explosion_detector_architecture.svg"

plt.savefig(png_path, dpi=220, bbox_inches='tight', facecolor='#F3F5F9', edgecolor='none')
plt.savefig(svg_path, bbox_inches='tight', facecolor='#F3F5F9', edgecolor='none')
plt.close(fig)

shutil.copy2(png_path, proj_dir / "acoustic_explosion_detector_architecture.png")
shutil.copy2(svg_path, proj_dir / "acoustic_explosion_detector_architecture.svg")

print(f"Successfully generated pristine poster:\n- Brain PNG: {png_path}\n- Brain SVG: {svg_path}\n- Workspace PNG: {proj_dir / 'acoustic_explosion_detector_architecture.png'}")
