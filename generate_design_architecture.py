"""
Generate System Design Architecture diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np

# Create design architecture diagram
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(8, 11.5, 'System Design Architecture - ML Diabetes Prediction Platform', 
        ha='center', va='top', fontsize=18, fontweight='bold')
ax.text(8, 11.1, 'Comprehensive system design with all layers, components, and integrations', 
        ha='center', va='top', fontsize=10, style='italic', color='#555555')

# ===== LAYER 1: PRESENTATION LAYER (Top) =====
layer1_y = 9.5
ax.text(0.3, layer1_y + 0.8, 'PRESENTATION LAYER', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2, pad=0.3))

# Web Interface
web_box = FancyBboxPatch((0.5, layer1_y), 2, 0.6, boxstyle="round,pad=0.05", 
                         edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(web_box)
ax.text(1.5, layer1_y+0.3, 'Web Portal', ha='center', va='center', fontsize=9, fontweight='bold')

# Mobile App
mobile_box = FancyBboxPatch((2.8, layer1_y), 2, 0.6, boxstyle="round,pad=0.05", 
                            edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(mobile_box)
ax.text(3.8, layer1_y+0.3, 'Mobile App', ha='center', va='center', fontsize=9, fontweight='bold')

# EHR Integration
ehr_box = FancyBboxPatch((5.1, layer1_y), 2, 0.6, boxstyle="round,pad=0.05", 
                         edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(ehr_box)
ax.text(6.1, layer1_y+0.3, 'EHR Integration', ha='center', va='center', fontsize=9, fontweight='bold')

# API Gateway
api_box = FancyBboxPatch((7.4, layer1_y), 2, 0.6, boxstyle="round,pad=0.05", 
                         edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(api_box)
ax.text(8.4, layer1_y+0.3, 'API Gateway', ha='center', va='center', fontsize=9, fontweight='bold')

# Admin Dashboard
admin_box = FancyBboxPatch((9.7, layer1_y), 2, 0.6, boxstyle="round,pad=0.05", 
                           edgecolor='#1976D2', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(admin_box)
ax.text(10.7, layer1_y+0.3, 'Admin Console', ha='center', va='center', fontsize=9, fontweight='bold')

# ===== LAYER 2: APPLICATION/BUSINESS LOGIC LAYER =====
layer2_y = 7.8
ax.text(0.3, layer2_y + 1.2, 'APPLICATION & BUSINESS LOGIC LAYER', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2, pad=0.3))

# User Management Service
user_service = FancyBboxPatch((0.5, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                              edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(user_service)
ax.text(1.4, layer2_y+0.8, 'User Mgmt', ha='center', va='center', fontsize=8, fontweight='bold')

# Authentication Service
auth_service = FancyBboxPatch((2.6, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                              edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(auth_service)
ax.text(3.5, layer2_y+0.8, 'Auth Service', ha='center', va='center', fontsize=8, fontweight='bold')

# Prediction Service
pred_service = FancyBboxPatch((4.7, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                              edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(pred_service)
ax.text(5.6, layer2_y+0.8, 'Prediction', ha='center', va='center', fontsize=8, fontweight='bold')

# Data Validation Service
validate_service = FancyBboxPatch((6.8, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                                  edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(validate_service)
ax.text(7.7, layer2_y+0.8, 'Validation', ha='center', va='center', fontsize=8, fontweight='bold')

# ML Pipeline Service
ml_service = FancyBboxPatch((8.9, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                            edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(ml_service)
ax.text(9.8, layer2_y+0.8, 'ML Pipeline', ha='center', va='center', fontsize=8, fontweight='bold')

# Audit Service
audit_service = FancyBboxPatch((11, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                               edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(audit_service)
ax.text(11.9, layer2_y+0.8, 'Audit Logs', ha='center', va='center', fontsize=8, fontweight='bold')

# Notification Service
notif_service = FancyBboxPatch((12.1, layer2_y+0.5), 1.8, 0.6, boxstyle="round,pad=0.05", 
                               edgecolor='#F57C00', facecolor='#FFF3E0', linewidth=2)
ax.add_patch(notif_service)
ax.text(13, layer2_y+0.8, 'Notifications', ha='center', va='center', fontsize=8, fontweight='bold')

# Core ML Model
model_box = FancyBboxPatch((4.2, layer2_y-0.8), 3.6, 0.7, boxstyle="round,pad=0.08", 
                           edgecolor='#C62828', facecolor='#FFEBEE', linewidth=3)
ax.add_patch(model_box)
ax.text(6, layer2_y-0.45, 'Random Forest ML Model (Trained & Optimized)', 
        ha='center', va='center', fontsize=9, fontweight='bold', color='#C62828')

# ===== LAYER 3: DATA & INTEGRATION LAYER =====
layer3_y = 5.2
ax.text(0.3, layer3_y + 1.2, 'DATA & INTEGRATION LAYER', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2, pad=0.3))

# Primary Database
db_box = FancyBboxPatch((0.5, layer3_y+0.4), 2.2, 0.7, boxstyle="round,pad=0.05", 
                        edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(db_box)
ax.text(1.6, layer3_y+0.75, 'PostgreSQL', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1.6, layer3_y+0.45, 'Primary DB', ha='center', va='center', fontsize=7)

# Cache Layer
cache_box = FancyBboxPatch((3.1, layer3_y+0.4), 2.2, 0.7, boxstyle="round,pad=0.05", 
                           edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(cache_box)
ax.text(4.2, layer3_y+0.75, 'Redis Cache', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(4.2, layer3_y+0.45, 'Performance', ha='center', va='center', fontsize=7)

# Message Queue
queue_box = FancyBboxPatch((5.7, layer3_y+0.4), 2.2, 0.7, boxstyle="round,pad=0.05", 
                           edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(queue_box)
ax.text(6.8, layer3_y+0.75, 'Message Queue', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(6.8, layer3_y+0.45, '(RabbitMQ)', ha='center', va='center', fontsize=7)

# Data Lake
lake_box = FancyBboxPatch((8.3, layer3_y+0.4), 2.2, 0.7, boxstyle="round,pad=0.05", 
                          edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(lake_box)
ax.text(9.4, layer3_y+0.75, 'Data Lake', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(9.4, layer3_y+0.45, '(S3 Storage)', ha='center', va='center', fontsize=7)

# ETL Pipeline
etl_box = FancyBboxPatch((10.9, layer3_y+0.4), 2.2, 0.7, boxstyle="round,pad=0.05", 
                         edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(etl_box)
ax.text(12, layer3_y+0.75, 'ETL Pipeline', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(12, layer3_y+0.45, 'Data Processing', ha='center', va='center', fontsize=7)

# External APIs
ext_box = FancyBboxPatch((13.5, layer3_y+0.4), 2, 0.7, boxstyle="round,pad=0.05", 
                         edgecolor='#388E3C', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(ext_box)
ax.text(14.5, layer3_y+0.75, 'External APIs', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(14.5, layer3_y+0.45, 'Lab Integration', ha='center', va='center', fontsize=7)

# Data Quality & Monitoring
quality_box = FancyBboxPatch((0.5, layer3_y-0.7), 15, 0.5, boxstyle="round,pad=0.05", 
                             edgecolor='#1565C0', facecolor='#BBDEFB', linewidth=2)
ax.add_patch(quality_box)
ax.text(8, layer3_y-0.45, 'Data Quality Checks, Schema Validation, Data Governance Framework', 
        ha='center', va='center', fontsize=8, fontweight='bold')

# ===== LAYER 4: INFRASTRUCTURE & DEPLOYMENT LAYER =====
layer4_y = 2.5
ax.text(0.3, layer4_y + 1.2, 'INFRASTRUCTURE & DEPLOYMENT LAYER', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2, pad=0.3))

# Docker Containers
docker_box = FancyBboxPatch((0.5, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                            edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(docker_box)
ax.text(1.5, layer4_y+0.8, 'Docker', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(1.5, layer4_y+0.45, 'Containers', ha='center', va='center', fontsize=7)

# Kubernetes Orchestration
k8s_box = FancyBboxPatch((2.8, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                         edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(k8s_box)
ax.text(3.8, layer4_y+0.8, 'Kubernetes', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(3.8, layer4_y+0.45, 'Orchestration', ha='center', va='center', fontsize=7)

# Load Balancer
lb_box = FancyBboxPatch((5.1, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                        edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(lb_box)
ax.text(6.1, layer4_y+0.8, 'Load', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(6.1, layer4_y+0.45, 'Balancer', ha='center', va='center', fontsize=7)

# Cloud Infrastructure
cloud_box = FancyBboxPatch((7.4, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                           edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(cloud_box)
ax.text(8.4, layer4_y+0.8, 'AWS/Azure', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(8.4, layer4_y+0.45, 'Cloud', ha='center', va='center', fontsize=7)

# Monitoring Stack
monitor_box = FancyBboxPatch((9.7, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                             edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(monitor_box)
ax.text(10.7, layer4_y+0.8, 'Prometheus', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(10.7, layer4_y+0.45, 'Monitoring', ha='center', va='center', fontsize=7)

# Logging Stack
log_box = FancyBboxPatch((12, layer4_y+0.4), 2, 0.65, boxstyle="round,pad=0.05", 
                         edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(log_box)
ax.text(13, layer4_y+0.8, 'ELK Stack', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(13, layer4_y+0.45, 'Logging', ha='center', va='center', fontsize=7)

# Security Layer
security_box = FancyBboxPatch((0.5, layer4_y-0.7), 15, 0.5, boxstyle="round,pad=0.05", 
                              edgecolor='#C62828', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(security_box)
ax.text(8, layer4_y-0.45, 'Security: TLS/SSL Encryption, RBAC, HIPAA Compliance, Audit Logging, Secret Management', 
        ha='center', va='center', fontsize=8, fontweight='bold')

# ===== CROSS-CUTTING CONCERNS =====
concerns_y = 0.3
ax.text(0.3, concerns_y + 0.5, 'CROSS-CUTTING CONCERNS', fontsize=11, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2, pad=0.3))

concerns_text = ['Security & Authorization', 'Error Handling & Recovery', 'Performance Optimization', 
                 'Scalability & Load Distribution', 'Monitoring & Alerting', 'Data Privacy & Compliance']
concern_x = 1.5
for i, concern in enumerate(concerns_text):
    if i % 3 == 0:
        concern_x = 1.5
        concerns_y -= 0.35
    ax.text(concern_x, concerns_y, f'• {concern}', fontsize=7.5)
    concern_x += 5

# Draw connections
# Layer 1 to Layer 2 (diagonal arrows)
for x_pos in [1.5, 3.8, 6.1, 8.4, 10.7]:
    arrow = FancyArrowPatch((x_pos, layer1_y), (x_pos, layer2_y+1.2), 
                           arrowstyle='->', mutation_scale=15, linewidth=1.5, 
                           color='#666666', alpha=0.6)
    ax.add_patch(arrow)

# Layer 2 to Layer 3
arrow23 = FancyArrowPatch((6, layer2_y-0.8), (6, layer3_y+1.2), 
                         arrowstyle='<->', mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow23)

# Layer 3 to Layer 4
arrow34 = FancyArrowPatch((8, layer3_y-0.7), (8, layer4_y+1.2), 
                         arrowstyle='<->', mutation_scale=20, linewidth=2, color='#333333')
ax.add_patch(arrow34)

# Add legend
legend_y = 10.5
ax.text(13.5, legend_y, 'Key Technologies:', fontsize=9, fontweight='bold')
ax.text(13.5, legend_y-0.35, '• Python/FastAPI', fontsize=7.5)
ax.text(13.5, legend_y-0.65, '• PostgreSQL/Redis', fontsize=7.5)
ax.text(13.5, legend_y-0.95, '• Docker/Kubernetes', fontsize=7.5)
ax.text(13.5, legend_y-1.25, '• Prometheus/ELK', fontsize=7.5)
ax.text(13.5, legend_y-1.55, '• AWS/Azure Cloud', fontsize=7.5)

plt.tight_layout()
plt.savefig('results/design_architecture.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✅ Design architecture diagram saved to: results/design_architecture.png")
plt.close()

print("\n✅ Design architecture diagram generated successfully!")
print("📊 File saved: results/design_architecture.png")
