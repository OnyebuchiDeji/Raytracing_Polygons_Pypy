import sys
import os
import pygame as pg
from PIL import Image
import moderngl as mgl
from tkinter.simpledialog import askstring


class Engine:
	def __init__(self, program=None, winWidth=1200, winHeight=675):
		pg.init()
		self.dimensions = [winWidth, winHeight]
		pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 4)
		pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
		pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)

		self.window = pg.display.set_mode(
			(winWidth, winHeight), flags=pg.OPENGL | pg.DOUBLEBUF)

		self.ctx = mgl.create_context()
		self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)

		self.clock = pg.time.Clock()
		self.time_elapsed = None
		self.delta_time = None
		self.prev_time = None
		self.fps = 60

		self.clear_color = (0, 0, 0)

		self.program = program


	def set_clear_color(self, col: tuple[int,int,int]):
		self.clear_color = col


	def onexit(self):
		self.program.onexit()
		pg.quit()
		sys.exit()
		exit()

	def poll_events(self):
		for e in pg.event.get():
			if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
				self.onexit()

	def save_pixels_as_png(self, imageFormat: str="png"):
		image_name = askstring("Image Name", "Save Image As", initialvalue="_")
		if not image_name:
			print("Cancelled. Not Saving Image")
			return

		save_dir = os.path.join(os.path.dirname(__file__), "..", "_scrnshots", image_name + f".{imageFormat}")

		if imageFormat == "png":
			# raw_px_data = pg.image.tobytes(self.window, "RGBA")
			raw_px_data = self.ctx.screen.read(components=4)
			image = Image.frombytes("RGBA", self.ctx.fbo.size, raw_px_data)
		else:
			# raw_px_data = pg.image.tobytes(self.window, "RGB")
			raw_px_data = self.ctx.screen.read(components=3)
			image = Image.frombytes("RGB", self.ctx.fbo.size, raw_px_data)

		image = image.transpose(Image.FLIP_TOP_BOTTOM)
		image.save(save_dir)
		print("Saved Image {} at: {}".format(image_name, save_dir))


	def update_caption(self):
		pg.display.set_caption(f"{self.program.title}, {self.program.current_subtitle} | fps: {self.clock.get_fps(): .4f}")

	def update(self):
		self.time_elapsed = pg.time.get_ticks() * 0.001
		self.delta_time = self.time_elapsed - self.prev_time if self.prev_time is not None else 0
		self.prev_time = self.time_elapsed
		self.update_caption()
		self.program.update()
		pg.display.flip()	#	must call update here even with moderngl since pygame is owner
		self.clock.tick(self.fps)

	def render(self):
		self.ctx.clear(color=self.clear_color)
		self.program.render()

	def start(self):
		while True:
			self.poll_events()
			self.update()
			self.render()