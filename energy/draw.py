import matplotlib.pyplot as plt 
from matplotlib.pyplot import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']


def draw(test_y, test_y_pred, test_y_add, font_size=20, title='', path=''):
    # generate a y=x reference line
    line_x = [0,max(test_y)]
    line_y = [0,max(test_y)]
    
    fig, ax = plt.subplots(1)
    plt.style.use('seaborn-paper')
    fig.set_size_inches(6, 5)
    ax.autoscale(True)
    plt.plot(line_x, line_y, color='#4169E1', linewidth=3, label='y=x')
    ax.scatter(test_y, test_y_pred, marker='o', alpha=0.7, color='#2a96a7', label='predicted')
    ax.scatter(test_y, test_y_add, marker='^', alpha=0.5, color='#FF6347', label='naive add')
    ax.set_xlabel("measured", fontsize=font_size)
    ax.set_ylabel("estimated", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', prop={'size':18}, bbox_to_anchor=(0.12,0.9), markerscale=3.0)
    # plt.tight_layout()
    plt.savefig(path)
    print("saved image " + path)
