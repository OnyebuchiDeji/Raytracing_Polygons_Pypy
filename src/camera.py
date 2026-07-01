from engine import pg
import glm


FOV = 100
NEAR = 0.1
FAR = 100
SPEED = 3.0
SENSITIVITY = 0.07

class FPSCamera:
	def __init__(self, dimensions, position=(0, 0, 4), yaw=-90, pitch=0):
		self.ar = dimensions[0] / dimensions[1]
		self.position = glm.vec3(position)
		self.up = glm.vec3(0, 1, 0)
		self.right = glm.vec3(1, 0, 0)
		self.forward = glm.vec3(0, 0, -1)
		self.yaw = yaw
		self.pitch = pitch
		self.fov = FOV
		self.fov_tangent = glm.tan(glm.radians(self.fov/2.0))

		self.view_mat = self.solve_view_matrix()
		# self.rot_mag = None

	def update(self, dt):
		self.move(dt)
		self.rotate()
		self.update_camera_vectors()

	def rotate(self):
		rel_x, rel_y = pg.mouse.get_rel()
		# self.rot_mag = vec2(rel_x, rel_y)
		# rel_x = rel_x + self.rot_mag.x * SPEED
		# rel_y = rel_y + self.rot_mag.y * SPEED
		self.yaw += rel_x * SENSITIVITY
		self.pitch -= rel_y * SENSITIVITY

		# self.rot_mag = max(0, self.rot_mag - self.rot_mag * dt)

	def update_camera_vectors(self):
		yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
		self.forward.x = glm.cos(yaw) * glm.cos(pitch)
		self.forward.y = glm.sin(pitch)
		self.forward.z = glm.sin(yaw) * glm.cos(pitch)

		self.forward = glm.normalize(self.forward)
		self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
		self.up = glm.normalize(glm.cross(self.right, self.forward))

		self.view_mat = self.solve_view_matrix()

	def move(self, dt):
		velocity = SPEED * dt
		keys = pg.key.get_pressed()
		if keys[pg.K_w]:
			self.position += self.forward * velocity
		if keys[pg.K_s]:
			self.position -= self.forward * velocity
		if keys[pg.K_a]:
			self.position -= self.right * velocity
		if keys[pg.K_d]:
			self.position += self.right * velocity
		if keys[pg.K_q]:
			self.position += self.up * velocity
		if keys[pg.K_e]:
			self.position -= self.up * velocity
		#	zoom-in, zoom-out
		if keys[pg.K_i]:
			self.fov -= SPEED
			self.fov_tangent = glm.tan(glm.radians(self.fov/2.0))
		if keys[pg.K_o]:
			self.fov += SPEED
			self.fov_tangent = glm.tan(glm.radians(self.fov/2.0))
		# print("Position: ", self.position)


	def solve_view_matrix(self):
		return glm.lookAt(self.position, self.position + self.forward, self.up)