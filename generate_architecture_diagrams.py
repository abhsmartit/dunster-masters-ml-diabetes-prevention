"""
Generate Proposed Deployment Architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'ML-Based Diabetes Prediction System: Proposed Deployment Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# Color scheme
color_input = '#E3F2FD'
color_processing = '#FFF3E0'
color_storage = '#F3E5F5'
color_output = '#E8F5E9'
color_monitoring = '#FCE4EC'

# Layer 1: Data Input Interface
box1 = FancyBboxPatch((0.3, 7.5), 2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#1976D2', facecolor=color_input, linewidth=2)
ax.add_patch(box1)
ax.text(1.3, 8.3, 'Data Input', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.3, 7.95, 'Interface', ha='center', va='center', fontsize=10)
ax.text(1.3, 7.6, '(EHR/Web Form)', ha='center', va='center', fontsize=8, style='italic')

# Layer 2: Clinical Dashboard
box2 = FancyBboxPatch((7.3, 7.5), 2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#388E3C', facecolor=color_output, linewidth=2)
ax.add_patch(box2)
ax.text(8.3, 8.3, 'Clinical', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.3, 7.95, 'Dashboard', ha='center', va='center', fontsize=10)
ax.text(8.3, 7.6, '(React/Vue)', ha='center', va='center', fontsize=8, style='italic')

# Layer 3: Data Validation
box3 = FancyBboxPatch((2.8, 5.8), 1.8, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#F57C00', facecolor=color_processing, linewidth=2)
ax.add_patch(box3)
ax.text(3.7, 6.6, 'Data', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3.7, 6.25, 'Validation', ha='center', va='center', fontsize=10)
ax.text(3.7, 5.9, '& Cleaning', ha='center', va='center', fontsize=8)

# Layer 4: Model API
box4 = FancyBboxPatch((5.2, 5.8), 1.8, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#C62828', facecolor=color_processing, linewidth=2)
ax.add_patch(box4)
ax.text(6.1, 6.6, 'Model', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6.1, 6.25, 'API (REST)', ha='center', va='center', fontsize=10)
ax.text(6.1, 5.9, 'Docker', ha='center', va='center', fontsize=8, style='italic')

# Layer 5: ML Model
box5 = FancyBboxPatch((5.2, 4.2), 1.8, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#D32F2F', facecolor=color_processing, linewidth=2)
ax.add_patch(box5)
ax.text(6.1, 5, 'Random Forest', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(6.1, 4.65, 'Model', ha='center', va='center', fontsize=10)
ax.text(6.1, 4.3, '(Trained)', ha='center', va='center', fontsize=8, style='italic')

# Layer 6: Database
box6 = FancyBboxPatch((2.8, 3.5), 1.8, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#7B1FA2', facecolor=color_storage, linewidth=2)
ax.add_patch(box6)
ax.text(3.7, 4.3, 'PostgreSQL', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3.7, 3.95, 'Database', ha='center', va='center', fontsize=10)
ax.text(3.7, 3.6, '(Audit Logs)', ha='center', va='center', fontsize=8, style='italic')

# Layer 7: Monitoring & Logging
box7 = FancyBboxPatch((0.3, 3.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#C2185B', facecolor=color_monitoring, linewidth=2)
ax.add_patch(box7)
ax.text(1.4, 4.3, 'Monitoring &', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.4, 3.95, 'Logging', ha='center', va='center', fontsize=10)
ax.text(1.4, 3.6, '(Prometheus)', ha='center', va='center', fontsize=8, style='italic')

# Layer 8: Continuous Learning
box8 = FancyBboxPatch((7.3, 3.5), 2.2, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#1565C0', facecolor=color_processing, linewidth=2)
ax.add_patch(box8)
ax.text(8.4, 4.3, 'Continuous', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(8.4, 3.95, 'Learning', ha='center', va='center', fontsize=10)
ax.text(8.4, 3.6, '(Retraining)', ha='center', va='center', fontsize=8, style='italic')

# Layer 9: Security & HIPAA
box9 = FancyBboxPatch((3.5, 1.5), 3, 1.2, boxstyle="round,pad=0.1", 
                       edgecolor='#00695C', facecolor=color_storage, linewidth=2)
ax.add_patch(box9)
ax.text(5, 2.35, 'Security & Compliance', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 1.95, 'HIPAA Encryption, RBAC, Audit Trail', ha='center', va='center', fontsize=9)

# Arrows - Main flow
# Input to Validation
arrow1 = FancyArrowPatch((1.3, 7.5), (3.7, 6.95), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow1)

# Input to Dashboard
arrow2 = FancyArrowPatch((2.3, 8.1), (7.3, 8.1), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#666666', linestyle='--')
ax.add_patch(arrow2)

# Validation to Model API
arrow3 = FancyArrowPatch((4.6, 6.2), (5.5, 6.2), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow3)

# Model API to ML Model
arrow4 = FancyArrowPatch((6.1, 5.8), (6.1, 5.4), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow4)

# ML Model to Dashboard
arrow5 = FancyArrowPatch((7, 4.8), (8, 7.5), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#666666', linestyle='--')
ax.add_patch(arrow5)

# Model to Database
arrow6 = FancyArrowPatch((5.2, 4.8), (4.6, 4.0), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow6)

# Monitoring
arrow7 = FancyArrowPatch((3.7, 3.5), (2.5, 3.5), arrowstyle='<->', 
                        mutation_scale=20, linewidth=2, color='#C2185B')
ax.add_patch(arrow7)

# Continuous Learning
arrow8 = FancyArrowPatch((6.1, 4.2), (8.4, 4.6), arrowstyle='->', 
                        mutation_scale=20, linewidth=2, color='#1565C0', linestyle=':')
ax.add_patch(arrow8)

# Security to all
arrow9 = FancyArrowPatch((4, 2.65), (5, 3.5), arrowstyle='-', 
                        mutation_scale=20, linewidth=1.5, color='#00695C', alpha=0.5)
ax.add_patch(arrow9)

# Add legends and annotations
legend_y = 0.8
ax.text(0.3, legend_y, 'Key Features:', fontsize=10, fontweight='bold')
ax.text(0.3, legend_y-0.35, '• RESTful API for scalability', fontsize=9)
ax.text(0.3, legend_y-0.65, '• Docker containerization', fontsize=9)
ax.text(0.3, legend_y-0.95, '• HIPAA-compliant security', fontsize=9)

ax.text(5, legend_y, 'Implementation Phases:', fontsize=10, fontweight='bold')
ax.text(5, legend_y-0.35, '• Phase 1: Silent Mode (Months 1-6)', fontsize=9)
ax.text(5, legend_y-0.65, '• Phase 2: Decision Support (Months 7-12)', fontsize=9)
ax.text(5, legend_y-0.95, '• Phase 3: Prospective Validation (Months 13-24)', fontsize=9)

ax.text(8, legend_y, 'Stack:', fontsize=10, fontweight='bold')
ax.text(8, legend_y-0.35, '• Backend: Python/FastAPI', fontsize=9)
ax.text(8, legend_y-0.65, '• DB: PostgreSQL', fontsize=9)
ax.text(8, legend_y-0.95, '• Orchestration: Kubernetes', fontsize=9)

plt.tight_layout()
plt.savefig('results/deployment_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Deployment Architecture diagram saved to: results/deployment_architecture.png")
plt.close()

# Create a detailed technical architecture diagram
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(5, 9.5, 'Technical Architecture & Data Flow', 
        ha='center', va='top', fontsize=16, fontweight='bold')

# User Tier
user_box = FancyBboxPatch((0.5, 8), 2, 0.8, boxstyle="round,pad=0.05", 
                          edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(user_box)
ax.text(1.5, 8.4, 'Clinician/User', ha='center', va='center', fontsize=10, fontweight='bold')

# Web Interface
web_box = FancyBboxPatch((4, 8), 2, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(web_box)
ax.text(5, 8.4, 'Web Interface', ha='center', va='center', fontsize=10, fontweight='bold')

# API Gateway
api_box = FancyBboxPatch((7.5, 8), 2, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(api_box)
ax.text(8.5, 8.4, 'API Gateway', ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows to next layer
for x_pos in [1.5, 5, 8.5]:
    arrow = FancyArrowPatch((x_pos, 8), (x_pos, 7.2), arrowstyle='->', 
                           mutation_scale=15, linewidth=1.5, color='#333333')
    ax.add_patch(arrow)

# Application Layer
app_box = FancyBboxPatch((2, 6.3), 6, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor='#C62828', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(app_box)
ax.text(5, 6.7, 'FastAPI/Flask Server (Containerized in Docker)', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows down
arrow_app1 = FancyArrowPatch((4, 6.3), (2.5, 5.5), arrowstyle='->', 
                            mutation_scale=15, linewidth=1.5, color='#333333')
ax.add_patch(arrow_app1)
arrow_app2 = FancyArrowPatch((6, 6.3), (7.5, 5.5), arrowstyle='->', 
                            mutation_scale=15, linewidth=1.5, color='#333333')
ax.add_patch(arrow_app2)

# ML Model Service
model_box = FancyBboxPatch((1.5, 4.5), 2.2, 0.9, boxstyle="round,pad=0.05", 
                           edgecolor='#D32F2F', facecolor='#FCE4EC', linewidth=2)
ax.add_patch(model_box)
ax.text(2.6, 5.1, 'Random Forest', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(2.6, 4.75, 'Model', ha='center', va='center', fontsize=9)

# Data Processing Service
process_box = FancyBboxPatch((6.3, 4.5), 2.2, 0.9, boxstyle="round,pad=0.05", 
                             edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(process_box)
ax.text(7.4, 5.1, 'Data', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(7.4, 4.75, 'Processing', ha='center', va='center', fontsize=9)

# Arrows down
arrow_down1 = FancyArrowPatch((2.6, 4.5), (2.6, 3.7), arrowstyle='->', 
                             mutation_scale=15, linewidth=1.5, color='#333333')
ax.add_patch(arrow_down1)
arrow_down2 = FancyArrowPatch((7.4, 4.5), (7.4, 3.7), arrowstyle='->', 
                             mutation_scale=15, linewidth=1.5, color='#333333')
ax.add_patch(arrow_down2)

# Data Layer
data_box = FancyBboxPatch((1, 2.8), 2.2, 0.8, boxstyle="round,pad=0.05", 
                          edgecolor='#1565C0', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(data_box)
ax.text(2.1, 3.2, 'PostgreSQL', ha='center', va='center', fontsize=9, fontweight='bold')

cache_box = FancyBboxPatch((6.8, 2.8), 2.2, 0.8, boxstyle="round,pad=0.05", 
                           edgecolor='#00695C', facecolor='#E0F2F1', linewidth=2)
ax.add_patch(cache_box)
ax.text(7.9, 3.2, 'Redis Cache', ha='center', va='center', fontsize=9, fontweight='bold')

# Monitoring
monitor_box = FancyBboxPatch((0.5, 1.3), 9, 1, boxstyle="round,pad=0.05", 
                             edgecolor='#C2185B', facecolor='#FCE4EC', linewidth=2)
ax.add_patch(monitor_box)
ax.text(1.5, 2, 'Prometheus', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(3.5, 2, 'Grafana', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5.5, 2, 'ELK Stack', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(7.5, 2, 'Alert Manager', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5, 1.5, 'Monitoring & Logging Infrastructure', ha='center', va='center', fontsize=8, style='italic')

# Arrows to monitoring
arrow_mon1 = FancyArrowPatch((2.1, 2.8), (2.5, 2.3), arrowstyle='-', 
                            mutation_scale=10, linewidth=1, color='#C2185B', alpha=0.6)
ax.add_patch(arrow_mon1)
arrow_mon2 = FancyArrowPatch((7.9, 2.8), (7.5, 2.3), arrowstyle='-', 
                            mutation_scale=10, linewidth=1, color='#C2185B', alpha=0.6)
ax.add_patch(arrow_mon2)

# Add description box
desc_box = FancyBboxPatch((0.3, 0.1), 9.4, 1, boxstyle="round,pad=0.08", 
                          edgecolor='#555555', facecolor='#FAFAFA', linewidth=1)
ax.add_patch(desc_box)
ax.text(5, 0.8, 'Deployment Environment: Kubernetes Cluster with Docker Containers', 
        ha='center', va='top', fontsize=9, fontweight='bold')
ax.text(5, 0.5, 'Scalability: Horizontal scaling through container replication | Security: TLS encryption, RBAC, secrets management', 
        ha='center', va='center', fontsize=8)
ax.text(5, 0.2, 'High Availability: Multi-instance setup with load balancing and failover', 
        ha='center', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('results/technical_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Technical Architecture diagram saved to: results/technical_architecture.png")
plt.close()

print("\n✅ Both architecture diagrams generated successfully!")
print("📊 Files saved:")
print("   1. results/deployment_architecture.png")
print("   2. results/technical_architecture.png")
