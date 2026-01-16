"""
Generate ML Pipeline Workflow diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Create workflow diagram
fig, ax = plt.subplots(1, 1, figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# Title
ax.text(8, 10.5, 'ML-Based Diabetes Prediction: Complete Workflow Diagram', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# Phase 1: Data Collection & Preparation (Left side)
phase1_y = 8.5
ax.text(1.5, 9.8, 'Phase 1: Data Collection & Preparation', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2, pad=0.5))

# Step 1.1: Data Collection
box1 = FancyBboxPatch((0.3, phase1_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box1)
ax.text(1.4, phase1_y+0.5, 'Patient Health Data', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.4, phase1_y+0.15, 'Collection', ha='center', va='center', fontsize=9)

# Step 1.2: Data Cleaning
box2 = FancyBboxPatch((3, phase1_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box2)
ax.text(4.1, phase1_y+0.5, 'Data Cleaning &', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(4.1, phase1_y+0.15, 'Preprocessing', ha='center', va='center', fontsize=9)

# Step 1.3: EDA
box3 = FancyBboxPatch((5.7, phase1_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box3)
ax.text(6.8, phase1_y+0.5, 'Exploratory Data', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(6.8, phase1_y+0.15, 'Analysis', ha='center', va='center', fontsize=9)

# Step 1.4: Feature Engineering
box4 = FancyBboxPatch((8.4, phase1_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(box4)
ax.text(9.5, phase1_y+0.5, 'Feature', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.5, phase1_y+0.15, 'Engineering', ha='center', va='center', fontsize=9)

# Arrows Phase 1
for i, x in enumerate([1.4, 4.1, 6.8]):
    arrow = FancyArrowPatch((x+1.1, phase1_y+0.4), (x+1.9, phase1_y+0.4), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#333333')
    ax.add_patch(arrow)

# Phase 2: Model Development (Middle)
phase2_y = 6.8
ax.text(1.5, 7.6, 'Phase 2: Model Development & Training', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2, pad=0.5))

# Step 2.1: Data Splitting
box5 = FancyBboxPatch((0.3, phase2_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(box5)
ax.text(1.4, phase2_y+0.5, 'Train-Test', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.4, phase2_y+0.15, 'Split (80-20)', ha='center', va='center', fontsize=9)

# Step 2.2: Algorithm Selection
box6 = FancyBboxPatch((3, phase2_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(box6)
ax.text(4.1, phase2_y+0.5, '9 ML Algorithms', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(4.1, phase2_y+0.15, 'Selected', ha='center', va='center', fontsize=9)

# Step 2.3: Model Training
box7 = FancyBboxPatch((5.7, phase2_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(box7)
ax.text(6.8, phase2_y+0.5, 'Model Training &', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(6.8, phase2_y+0.15, 'Hyperparameter Tuning', ha='center', va='center', fontsize=9)

# Step 2.4: Best Model Selection
box8 = FancyBboxPatch((8.4, phase2_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#C62828', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(box8)
ax.text(9.5, phase2_y+0.5, 'Best Model', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.5, phase2_y+0.15, 'Selection', ha='center', va='center', fontsize=9)

# Arrows Phase 2
for i, x in enumerate([1.4, 4.1, 6.8]):
    arrow = FancyArrowPatch((x+1.1, phase2_y+0.4), (x+1.9, phase2_y+0.4), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#333333')
    ax.add_patch(arrow)

# Phase 3: Model Evaluation
phase3_y = 5.1
ax.text(1.5, 5.9, 'Phase 3: Model Evaluation & Validation', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2, pad=0.5))

# Step 3.1: Performance Metrics
box9 = FancyBboxPatch((0.3, phase3_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                      edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box9)
ax.text(1.4, phase3_y+0.5, 'Calculate', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(1.4, phase3_y+0.15, 'Performance Metrics', ha='center', va='center', fontsize=9)

# Step 3.2: Feature Importance
box10 = FancyBboxPatch((3, phase3_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box10)
ax.text(4.1, phase3_y+0.5, 'Feature Importance', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(4.1, phase3_y+0.15, 'Analysis', ha='center', va='center', fontsize=9)

# Step 3.3: Cross-Validation
box11 = FancyBboxPatch((5.7, phase3_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box11)
ax.text(6.8, phase3_y+0.5, 'Cross-Validation', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(6.8, phase3_y+0.15, '(K-Fold)', ha='center', va='center', fontsize=9)

# Step 3.4: Model Validation
box12 = FancyBboxPatch((8.4, phase3_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(box12)
ax.text(9.5, phase3_y+0.5, 'Model Validation', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.5, phase3_y+0.15, 'on Test Data', ha='center', va='center', fontsize=9)

# Arrows Phase 3
for i, x in enumerate([1.4, 4.1, 6.8]):
    arrow = FancyArrowPatch((x+1.1, phase3_y+0.4), (x+1.9, phase3_y+0.4), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#333333')
    ax.add_patch(arrow)

# Phase 4: Deployment & Monitoring (Right side)
phase4_y = 3.4
ax.text(11.5, 4.2, 'Phase 4: Deployment & Monitoring', 
        fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2, pad=0.5))

# Step 4.1: Model Deployment
box13 = FancyBboxPatch((11.3, phase4_y), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(box13)
ax.text(12.4, phase4_y+0.5, 'Model Deployment', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(12.4, phase4_y+0.15, '(Docker/Production)', ha='center', va='center', fontsize=9)

# Step 4.2: Real-world Predictions
box14 = FancyBboxPatch((11.3, phase4_y-1.2), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(box14)
ax.text(12.4, phase4_y-0.7, 'Real-World', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(12.4, phase4_y-1.05, 'Predictions', ha='center', va='center', fontsize=9)

# Step 4.3: Performance Monitoring
box15 = FancyBboxPatch((11.3, phase4_y-2.4), 2.2, 0.8, boxstyle="round,pad=0.05", 
                       edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(box15)
ax.text(12.4, phase4_y-1.9, 'Monitoring &', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(12.4, phase4_y-2.25, 'Performance Tracking', ha='center', va='center', fontsize=9)

# Vertical arrows Phase 4
arrow_p4_1 = FancyArrowPatch((12.4, phase4_y), (12.4, phase4_y-0.4), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow_p4_1)
arrow_p4_2 = FancyArrowPatch((12.4, phase4_y-1.2), (12.4, phase4_y-1.6), 
                             arrowstyle='->', mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow_p4_2)

# Connect phases with main flow arrows
# Phase 1 to Phase 2
arrow_12 = FancyArrowPatch((9.5, phase1_y-0.2), (9.5, phase2_y+0.8), 
                          arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#555555')
ax.add_patch(arrow_12)

# Phase 2 to Phase 3
arrow_23 = FancyArrowPatch((9.5, phase2_y-0.2), (9.5, phase3_y+0.8), 
                          arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#555555')
ax.add_patch(arrow_23)

# Phase 3 to Phase 4 (right side connection)
arrow_34 = FancyArrowPatch((10.6, phase3_y+0.4), (11.3, phase4_y+0.4), 
                          arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#555555')
ax.add_patch(arrow_34)

# Feedback loop: Monitoring back to training
feedback_arrow = FancyArrowPatch((11.3, phase4_y-2), (8.5, phase2_y-0.3), 
                               arrowstyle='->', mutation_scale=20, linewidth=2, 
                               color='#D32F2F', linestyle='--', alpha=0.7)
ax.add_patch(feedback_arrow)
ax.text(9, 0.8, 'Continuous Improvement Loop', fontsize=9, style='italic', 
        color='#D32F2F', ha='center', fontweight='bold')

# Decision point after evaluation
decision_circle = Circle((6.8, 3.5), 0.3, edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(decision_circle)
ax.text(6.8, 3.5, '✓', ha='center', va='center', fontsize=16, color='#D32F2F', fontweight='bold')

# Result box
result_box = FancyBboxPatch((3.5, 0.8), 6.6, 1, boxstyle="round,pad=0.1", 
                           edgecolor='#1B5E20', facecolor='#E8F5E9', linewidth=3)
ax.add_patch(result_box)
ax.text(6.8, 1.55, 'Final Outcome: Production-Ready ML Model', ha='center', va='top', 
        fontsize=11, fontweight='bold', color='#1B5E20')
ax.text(6.8, 1.15, 'Random Forest Accuracy: 75.97% | ROC-AUC: 81.47% | Deployed with monitoring & feedback loop', 
        ha='center', va='center', fontsize=9, color='#1B5E20')

# Add metrics table on side
table_y = 1.5
ax.text(0.3, table_y+0.8, 'Key Metrics:', fontsize=10, fontweight='bold')
metrics = [
    'Accuracy: 75.97%',
    'Precision: 82.05%',
    'Recall: 59.26%',
    'F1-Score: 68.82%',
    'ROC-AUC: 81.47%'
]
for i, metric in enumerate(metrics):
    ax.text(0.3, table_y+0.4-i*0.3, f'• {metric}', fontsize=8)

plt.tight_layout()
plt.savefig('results/workflow_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Workflow diagram saved to: results/workflow_diagram.png")
plt.close()

# Create detailed implementation workflow
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(7, 9.5, 'Clinical Implementation Workflow: Deployment Phases', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# Phase 1: Silent Mode
phase_height = 2.2
box_w = 3
box_h = 1.5

# Silent Mode
silent_box = FancyBboxPatch((0.5, 6.5), box_w, box_h, boxstyle="round,pad=0.1", 
                            edgecolor='#FFA726', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(silent_box)
ax.text(2, 7.6, 'Phase 1:', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2, 7.3, 'Silent Mode', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2, 6.95, '(Months 1-6)', ha='center', va='center', fontsize=8, style='italic')
ax.text(2, 6.6, 'Algorithm runs alongside', ha='center', va='center', fontsize=8)

# Decision Support
support_box = FancyBboxPatch((4.2, 6.5), box_w, box_h, boxstyle="round,pad=0.1", 
                             edgecolor='#29B6F6', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(support_box)
ax.text(5.7, 7.6, 'Phase 2:', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5.7, 7.3, 'Decision Support', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(5.7, 6.95, '(Months 7-12)', ha='center', va='center', fontsize=8, style='italic')
ax.text(5.7, 6.6, 'Clinicians receive', ha='center', va='center', fontsize=8)

# Validation
validation_box = FancyBboxPatch((7.9, 6.5), box_w, box_h, boxstyle="round,pad=0.1", 
                                edgecolor='#66BB6A', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(validation_box)
ax.text(9.4, 7.6, 'Phase 3:', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.4, 7.3, 'Prospective Validation', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(9.4, 6.95, '(Months 13-24)', ha='center', va='center', fontsize=8, style='italic')
ax.text(9.4, 6.6, 'RCT comparing outcomes', ha='center', va='center', fontsize=8)

# Scaled Deployment
scaled_box = FancyBboxPatch((11.6, 6.5), box_w, box_h, boxstyle="round,pad=0.1", 
                            edgecolor='#AB47BC', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(scaled_box)
ax.text(13.1, 7.6, 'Phase 4:', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(13.1, 7.3, 'Scaled Deployment', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(13.1, 6.95, '(Month 25+)', ha='center', va='center', fontsize=8, style='italic')
ax.text(13.1, 6.6, 'Broad implementation', ha='center', va='center', fontsize=8)

# Arrows between phases
for i in range(3):
    x_start = 0.5 + box_w + (i * 3.7)
    arrow = FancyArrowPatch((x_start+0.1, 7.2), (x_start+3.5, 7.2), 
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#333333')
    ax.add_patch(arrow)

# Details below each phase
detail_y_start = 5.8

details = [
    ['• Pilot sites (2-3)', 
     '• No clinical impact',
     '• Performance tracking',
     '• Clinician feedback'],
    
    ['• Recommendations only',
     '• Clinician decides',
     '• Track concordance',
     '• Iterative refinement'],
    
    ['• Intervention group',
     '• Control group',
     '• Primary outcome',
     '• Cost-effectiveness'],
    
    ['• EHR integration',
     '• Training & certification',
     '• Governance setup',
     '• Ongoing monitoring']
]

for phase_idx, detail_list in enumerate(details):
    x_pos = 0.5 + (phase_idx * 3.7) + box_w/2
    for line_idx, detail in enumerate(detail_list):
        ax.text(x_pos, detail_y_start - line_idx*0.3, detail, 
               ha='center', va='top', fontsize=7.5)

# Success metrics
metrics_box = FancyBboxPatch((1, 1.5), 12, 3.2, boxstyle="round,pad=0.15", 
                             edgecolor='#1B5E20', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(metrics_box)

ax.text(7, 4.5, 'Success Criteria & Evaluation Metrics', ha='center', va='top', 
        fontsize=11, fontweight='bold', color='#1B5E20')

# Left column
ax.text(2.5, 4, 'Technical Metrics:', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(2.5, 3.7, '• Model Accuracy > 75%', ha='center', va='top', fontsize=8)
ax.text(2.5, 3.4, '• ROC-AUC > 0.80', ha='center', va='top', fontsize=8)
ax.text(2.5, 3.1, '• API Response Time < 100ms', ha='center', va='top', fontsize=8)
ax.text(2.5, 2.8, '• System Uptime > 99.9%', ha='center', va='top', fontsize=8)

# Middle column
ax.text(7, 4, 'Clinical Metrics:', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(7, 3.7, '• Diagnostic Sensitivity > 75%', ha='center', va='top', fontsize=8)
ax.text(7, 3.4, '• Positive Predictive Value > 80%', ha='center', va='top', fontsize=8)
ax.text(7, 3.1, '• Time to Diagnosis reduction', ha='center', va='top', fontsize=8)
ax.text(7, 2.8, '• Patient outcomes improvement', ha='center', va='top', fontsize=8)

# Right column
ax.text(11.5, 4, 'Implementation Metrics:', ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(11.5, 3.7, '• Clinician adoption rate > 80%', ha='center', va='top', fontsize=8)
ax.text(11.5, 3.4, '• User satisfaction > 4/5', ha='center', va='top', fontsize=8)
ax.text(11.5, 3.1, '• Cost per diagnosis reduction', ha='center', va='top', fontsize=8)
ax.text(11.5, 2.8, '• Minimal bias across groups', ha='center', va='top', fontsize=8)

# Final outcome
outcome_box = FancyBboxPatch((1.5, 0.2), 11, 1, boxstyle="round,pad=0.08", 
                            edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(outcome_box)
ax.text(7, 0.9, '✓ Production-Ready System with Continuous Monitoring & Improvement', 
       ha='center', va='center', fontsize=10, fontweight='bold', color='#D32F2F')

plt.tight_layout()
plt.savefig('results/implementation_phases.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Implementation phases diagram saved to: results/implementation_phases.png")
plt.close()

print("\n✅ All workflow diagrams generated successfully!")
print("📊 Files saved:")
print("   1. results/workflow_diagram.png - Complete ML pipeline workflow")
print("   2. results/implementation_phases.png - Clinical implementation phases")
