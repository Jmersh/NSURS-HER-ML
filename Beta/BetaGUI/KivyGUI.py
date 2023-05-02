# Beta GUI app based on Kivy for cross-platform machine learning

from screen1.screen1 import Screen1
from screen2.screen2 import Screen2
from screen3.screen3 import Screen3
from multitab_screen.multitab_screen import MultitabScreen
from navigation_drawer.navigation_drawer import ContentNavigationDrawer

from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager


class WindowManager(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(Screen1())
        self.add_widget(Screen2())
        self.add_widget(Screen3())
        self.add_widget(MultitabScreen(name='multitab_screen'))


class MyApp(MDApp):
    def build(self):
        return Builder.load_file('main.kv')


if __name__ == "__main__":
    MyApp().run()