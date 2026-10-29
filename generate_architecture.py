import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, width, height, text, facecolor='#E3F2FD', edgecolor='#1565C0', text_color='black', fontsize=10):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, horizontalalignment='center', verticalalignment='center', 
            fontsize=fontsize, weight='bold', color=text_color, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    # Using ax.annotate to draw arrows
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="->", lw=2.5, color='#78909C'), zorder=1)

# Base layout dimensions
w, h = 1.6, 1.2
x_gap = 2.2
y_pos = [8, 5, 2] # Facial, Audio, Text rows

# Row 1: Facial
draw_box(ax, 0.5, y_pos[0], w, h, "Image Frame\n(Facial)")
draw_box(ax, 0.5 + x_gap, y_pos[0], w, h, "Face Detection\n& Cropping\n(dlib)")
draw_box(ax, 0.5 + x_gap*2, y_pos[0], w, h, "XCeption CNN\nModel", facecolor='#FFF9C4', edgecolor='#F57F17')
draw_box(ax, 0.5 + x_gap*3, y_pos[0], w, h, "7-Class Probabilities\n(Facial)")

# Row 2: Speech
draw_box(ax, 0.5, y_pos[1], w, h, "Audio Stream\n(Speech)")
draw_box(ax, 0.5 + x_gap, y_pos[1], w, h, "MFCC Feature\nExtraction\n(librosa)")
draw_box(ax, 0.5 + x_gap*2, y_pos[1], w, h, "LSTM Network\nModel", facecolor='#FFF9C4', edgecolor='#F57F17')
draw_box(ax, 0.5 + x_gap*3, y_pos[1], w, h, "7-Class Probabilities\n(Speech)")

# Row 3: Text
draw_box(ax, 0.5, y_pos[2], w, h, "Transcribed Text\n(Linguistic)")
draw_box(ax, 0.5 + x_gap, y_pos[2], w, h, "Tokenization\n& Padding\n(BERT Tokenizer)")
draw_box(ax, 0.5 + x_gap*2, y_pos[2], w, h, "BERT\nModel", facecolor='#FFF9C4', edgecolor='#F57F17')
draw_box(ax, 0.5 + x_gap*3, y_pos[2], w, h, "7-Class Probabilities\n(Text)")

# Horizontal Arrows
for i in range(3):
    y = y_pos[i] + h/2
    draw_arrow(ax, 0.5+w, y, 0.5+x_gap, y)
    draw_arrow(ax, 0.5+x_gap+w, y, 0.5+x_gap*2, y)
    draw_arrow(ax, 0.5+x_gap*2+w, y, 0.5+x_gap*3, y)

# Fusion Block
fusion_x = 0.5 + x_gap*4
fusion_y = 4.5
fusion_w = 2.0
fusion_h = 2.2
draw_box(ax, fusion_x, fusion_y, fusion_w, fusion_h, "Majority Voting\nFusion\n\n(Confidence\nTie-breaker)", 
         facecolor='#E8F5E9', edgecolor='#2E7D32', fontsize=12)

# Arrows to Fusion
draw_arrow(ax, 0.5+x_gap*3+w, y_pos[0]+h/2, fusion_x, fusion_y+fusion_h-0.3)
draw_arrow(ax, 0.5+x_gap*3+w, y_pos[1]+h/2, fusion_x, fusion_y+fusion_h/2)
draw_arrow(ax, 0.5+x_gap*3+w, y_pos[2]+h/2, fusion_x, fusion_y+0.3)

# Final Output Block
out_x = fusion_x + 0.2
out_y = 1.5
out_w = 1.6
out_h = 1.2
draw_box(ax, out_x, out_y, out_w, out_h, "Final Emotion\nVerdict", facecolor='#FFCDD2', edgecolor='#C62828', fontsize=12)

# Arrow from Fusion to Output
draw_arrow(ax, fusion_x + fusion_w/2, fusion_y, out_x + out_w/2, out_y + out_h)

# Title
plt.suptitle("EmotiSense: Multimodal Emotion Recognition Architecture", fontsize=18, weight='bold', y=1.05)
plt.title("A parallel processing framework fusing visual, acoustic, and linguistic modalities", fontsize=12, style='italic', pad=15)

plt.tight_layout()
plt.savefig("model_architecture.png", dpi=300, bbox_inches='tight')
print("Architecture image saved to model_architecture.png")
