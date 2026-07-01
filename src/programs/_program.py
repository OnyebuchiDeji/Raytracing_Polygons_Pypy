
from engine import pg

class Program:
	def __init__(self, title="Program", description="", engine=None, demonstrations=None):
		self.title = title
		self.description = description
		self.demonstrations = demonstrations
		self.current_demonstration = 0
		self.current_subtitle = None
		self.with_texture = False
		self.vertex_data = None
		self.vertex_format = "2f" 	#	just x y
		self.attributes = ["a_VertexPosition"] 		#	names of structures 
		self.shader_program = None
		self.vbo = None
		self.vao = None
		self.mesh = None
		self.bvh = None
		self.camera = None
		self.engine = engine
		self.could_throw = {}
		self.oninit()


	def __repr__(self):
		return (
			f"Program(title:{self.title}, "
			f"description:{self.description}"
		)

	def oninit(self):
		"""Perform every necessary setup before upon initialization"""
		...

	def update(self):
		"""Evaluates operations that change every frame"""
		...

	def render(self):
		"""Called after update. Performs operations directly linked to rendering of shaders"""
		...

	def onexit(self):
		"""Called upon exit to free resources"""
		...

	def run(self):
		self.engine.start()


	def sense_demonstration_change(self):
		keys = pg.key.get_pressed()
		sense_key_codes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 

		if keys[pg.K_m]:
			self.engine.save_pixels_as_png()
		
		for key_code in sense_key_codes:
			# print("Key code: ", key_code)
			# key code must be up to the number of demonstrations
			if keys[pg.key.key_code(key_code)]:
				if int(key_code) > len(self.demonstrations) - 1:
					return
				if self.current_demonstration == int(key_code):
					return
				self.onexit()
				self.current_demonstration = int(key_code)
				self.oninit()