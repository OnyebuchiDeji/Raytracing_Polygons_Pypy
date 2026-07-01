from engine import pg
import glm

SPEED = 1.2

class Model:
	def __init__(
		self, position=(0, 0, 0),
		orientation=(0, 0, 0), scale=(1, 1, 1)
	):
		self.position = glm.vec3(position)
		self.orientation = glm.vec3([glm.radians(a) for a in orientation])
		self.scale = glm.vec3(scale)
		self.mat = glm.mat4()


	def solve_matrix(self):
		self.mat = glm.mat4()
		# Translate
		self.mat = glm.translate(self.mat, self.position)
		# Rotate
		self.mat = glm.rotate(self.mat, self.orientation.x, glm.vec3(0, 0, 1))
		self.mat = glm.rotate(self.mat, self.orientation.y, glm.vec3(0, 1, 0))
		self.mat = glm.rotate(self.mat, self.orientation.z, glm.vec3(1, 0, 0))
		#   Scale
		self.mat = glm.scale(self.mat, self.scale)

	def get_mat_inverse(self):
		return glm.inverse(self.mat)


	def update(self, dt:float):
		self.move(dt)
		self.solve_matrix()

	def move(self, dt:float):
		velocity = SPEED * dt
		keys = pg.key.get_pressed()
		if keys[pg.K_UP]:
			self.position.y += velocity
		if keys[pg.K_DOWN]:
			self.position.y -= velocity
		if keys[pg.K_LEFT]:
			self.position.x -= velocity
		if keys[pg.K_RIGHT]:
			self.position.x += velocity