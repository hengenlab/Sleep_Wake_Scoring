import math
class Cursor(object):
    def __init__(self, ax1, ax2, ax3):
        self.clicked=False
        self.movie_mode = False
        self.second_click = False
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.movie_mode = False
        self.bins = []
        self.change_bins = False
        self.movie_bin = 0
        self.DONE = False
        print('making a cursor')


    def on_move(self, event):
        print('on move')

    def on_press(self, event):
        if event.key == 'd':
            print('DONE SCORING')
            self.DONE = True


    def in_axes(self, event):
        if event.inaxes == self.ax3:
            self.movie_mode = True
            print('MOVIE MODE!')
        else:
            self.movie_mode = False
    def pull_up_movie(self, event):
        print('gon pull up some movies')


    def on_click(self, event):
        if self.movie_mode:
            self.movie_bin = event.xdata
            print(f'video bin (xdata): {event.xdata}')
            print(f'x: {event.x}')
        elif self.clicked:
            if event.inaxes != self.ax2:
                print('please click in the second figure to select bins')
            else:
                print(F'SECOND CLICK ----  xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                self.bins.append(math.floor(event.xdata))
                self.clicked = False
                self.change_bins = True
        else:
            if event.inaxes != self.ax2:
                print('please click in the second figure to select bins')
            else:
                self.bins.append(math.floor(event.xdata))
                print(f'FIRST CLICK ----- xdata:{event.xdata} x:{event.x} axes: {event.inaxes}')
                self.clicked = True
