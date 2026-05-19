import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 0. 全局字体与样式配置 =================
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial'] # 全局锁定 Arial
plt.rcParams['axes.unicode_minus'] = False 

# ================= 1. 数据读取与计算 =================
excel_path = r'D:\VOCs钢铁\数据产出\kgdemoup\泛化独立评估.xlsx'
print(f"正在读取数据: {excel_path} ...")
df = pd.read_excel(excel_path)
experts = ['Expert A', 'Expert B', 'Expert C', 'Expert D']
df_scores = df[experts].apply(pd.to_numeric, errors='coerce').dropna()

results = []
for exp in experts:
    scores = df_scores[exp]
    tp = (scores == 2).sum()
    fn = (scores == 1).sum()
    fp = (scores == 0).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results.append({'Expert': exp, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'TP': tp, 'FN': fn, 'FP': fp})

# 计算总体 (Overall)
all_scores = df_scores.values.flatten()
tp_all = (all_scores == 2).sum()
fn_all = (all_scores == 1).sum()
fp_all = (all_scores == 0).sum()
prec_all = tp_all / (tp_all + fp_all)
rec_all = tp_all / (tp_all + fn_all)
f1_all = 2 * prec_all * rec_all / (prec_all + rec_all)
results.append({'Expert': 'Overall', 'Precision': prec_all, 'Recall': rec_all, 'F1-Score': f1_all, 'TP': tp_all, 'FN': fn_all, 'FP': fp_all})

res_df = pd.DataFrame(results)

# ================= 2. 准备绘图数据 =================
# (a) 性能指标
metrics_df = res_df.set_index('Expert')[['Precision', 'Recall', 'F1-Score']]

# (b) TP/FN/FP 数量分布 (排除 Overall)
conf_df = res_df.set_index('Expert')[['TP', 'FN', 'FP']].iloc[:-1] 

# (c) 全景打分矩阵
raw_scores_t = df_scores.T

# ================= 3. 绘制组合大图 (GridSpec 高精度布局) =================
print("正在绘制 (a)(b)(c) 组合大图...")

fig = plt.figure(figsize=(16, 11))

# 采用 2行 100列的网格
gs = fig.add_gridspec(2, 100, height_ratios=[1.2, 1])

# --------------------------------------------------------
# --- 绘制 (a) 性能指标热力图 (左上角) ---
# --------------------------------------------------------

ax1 = fig.add_subplot(gs[0, :50])
sns.heatmap(metrics_df, annot=True, cmap="RdBu", fmt=".3f", vmin=0.5, vmax=1.0, 
            annot_kws={"size": 13, "weight": "bold", "family": "Arial"}, 
            cbar_kws={'label': 'Score (0 to 1)'}, ax=ax1)

ax1.set_title("(a)", loc='left', fontsize=20, fontweight='bold', pad=15, fontfamily='Arial')
ax1.set_ylabel("Expert", fontsize=14, fontweight='bold', fontfamily='Arial')
ax1.tick_params(axis='y', rotation=0, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# --------------------------------------------------------
# --- 绘制 (b) 真假阳性分布热力图 (右上角) ---
# --------------------------------------------------------

ax2 = fig.add_subplot(gs[0, 55:])
sns.heatmap(conf_df, annot=True, cmap="Blues", fmt="g", 
            annot_kws={"size": 14, "weight": "bold", "family": "Arial"}, ax=ax2)

ax2.set_title("(b)", loc='left', fontsize=20, fontweight='bold', pad=15, fontfamily='Arial')
ax2.set_ylabel("", fontsize=14) 
ax2.tick_params(axis='y', rotation=0, labelsize=12)
ax2.tick_params(axis='x', labelsize=12)

# --------------------------------------------------------
# --- 绘制 (c) 专家一致性全景图 (底部) ---
# --------------------------------------------------------
# 让图C从第 4 列开始（即向右平移 4%）
ax3 = fig.add_subplot(gs[1, 4:100]) 

# 使用经典的离散学术色盘: 砖红(FP), 灰蓝(FN), 深蓝(TP)
cmap_custom = sns.color_palette(["#B2182B", "#D1E5F0", "#2166AC"]) 

# 让色条更贴近图表主体
sns.heatmap(raw_scores_t, cmap=cmap_custom, vmin=-0.5, vmax=2.5, 
            linewidths=0.8, linecolor='white', ax=ax3,
            cbar_kws={"pad": 0.015})

# 设置清晰的图例标签
cbar = ax3.collections[0].colorbar
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['0 (FP)', '1 (FN)', '2 (TP)'], fontsize=12, fontfamily='Arial')

ax3.set_title("(c)", loc='left', fontsize=20, fontweight='bold', pad=15, fontfamily='Arial')
ax3.set_xlabel("Question ID", fontsize=14, fontweight='bold', fontfamily='Arial')
ax3.set_ylabel("Expert", fontsize=14, fontweight='bold', fontfamily='Arial')
ax3.tick_params(axis='y', rotation=0, labelsize=12)
ax3.tick_params(axis='x', labelsize=8) 

# 💡 核心修改：移除紧凑布局，改用 GridSpec 的自适应更新，防止其再次强行压缩内部间距
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# ================= 4. 保存图片 =================
save_dir = r'D:\VOCs钢铁\数据产出\kgdemoup\demo1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "Figure_Expert_Evaluation.png")

# 使用 bbox_inches='tight' 可以保证保存时不会切掉边缘的标签
plt.savefig(save_path, dpi=400, bbox_inches='tight')
print(f"✅ 完美！极简学术风的高清组合大图已保存至：\n{save_path}")

# ================= 4. 保存图片 =================
save_dir = r'D:\VOCs钢铁\数据产出\kgdemoup\demo1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, "Figure_Expert_Evaluation.png")

plt.savefig(save_path, dpi=400, bbox_inches='tight')
print(f"✅ 完美！极简学术风的高清组合大图已保存至：\n{save_path}")