import matplotlib.pyplot as plt 
from matplotlib.pyplot import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']


def draw(test_y, test_y_pred, font_size=10, title='', path=''):
    # generate a y=x reference line
    line_x = [0,max(test_y)]
    line_y = [0,max(test_y)]
    
    fig, ax = plt.subplots(1)
    plt.style.use('seaborn-paper')
    fig.set_size_inches(8, 5)
    ax.autoscale(True)
    ax.scatter(test_y, test_y_pred, marker='o', alpha=0.6)
    plt.plot(line_x, line_y, color='r')
    font_size = 10
    ax.set_xlabel("measured", fontsize=font_size)
    ax.set_ylabel("estimated", fontsize=font_size)
    ax.set_title(title, fontsize=font_size)
    plt.savefig(path)
    print("saved image " + path)