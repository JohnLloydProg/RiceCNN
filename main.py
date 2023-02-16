from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.lang import Builder
from kivy.config import Config
from kivy.graphics.texture import Texture
import numpy as np
import settings
import tensorflow
import cam
import cv2

Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'width', '480')
Config.set('graphics', 'height', '320')

Builder.load_file('Main.kv')

class MainScreen(Screen):
    def on_enter(self, *args):
        self.ret = None
        self.app = App.get_running_app()
        self.url = ""
        self.stream = Clock.schedule_interval(self.update_stream, 0.3)
        if self.url:
            cam.set_resolution(self.url, index=8)
        return super().on_enter(*args)
    def update_stream(self, *args):
        if self.url:
            try:
                if self.cap.isOpened():
                    self.ret, self.frame = self.cap.read()
                    self.frame_stream = cv2.resize(self.frame, (settings.picture_width, settings.picture_height))
                    buffer = cv2.flip(self.frame_stream, 0).tobytes()
                    texture = Texture.create(size=(self.frame_stream.shape[1], self.frame_stream.shape[0]), colorfmt='bgr')
                    texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
                    self.ids['stream'].texture = texture
                self.frame = self.frame/255.0
            except:
                pass
    def update_ip_address(self, textinput):
        self.url = textinput.text
        if self.url:
            self.cap = cv2.VideoCapture(self.url + ":81/stream")
            cam.set_resolution(self.url, index=8)
    def check(self):
        print('checking')
        if self.ret:
            disease_result = self.app.model.predict(self.frame.reshape(-1, settings.picture_height, settings.picture_width, 3))
            y = np.argmax(disease_result, axis=1)
            if y[0] == 0:
                self.ids['label'].text = 'Bacterial Leaf Streak - Streak'
            elif y[0] == 1:
                self.ids['label'].text = 'Stem Rot - Rot'
            else:
                self.ids['label'].text = 'Sheath Blight - Spot'
            print(y[0])
            


class MainApp(App):
    sm = ScreenManager()
    model = tensorflow.keras.models.load_model('./logs/model.h5')
    print(model.summary())

    def build(self):
        main_screen = MainScreen(name='main screen')
        self.sm.add_widget(main_screen)
        self.sm.current = 'main screen'
        return self.sm

if __name__ == '__main__':
    MainApp().run()
