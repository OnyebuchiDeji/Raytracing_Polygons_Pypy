"""Entry point of project"""
from programs.program1 import Program1
from programs.program2 import Program2
from programs.program3 import Program3
from programs.program4 import Program4
from programs.program5 import Program5


class App:
	def __init__(self):
		self.programs = {
			# 'raytace_no_optimization': Program1(),
			# 'raytace_global_aabb_optimization': Program2(),
			# 'raytace_bvh_optimization_scrapped': Program3(),
			# 'raytace_bvh_optimization_issues': Program4(),
			'raytace_bvh_optimization_fixed': Program5(),
		}

	def run(self):
		# self.programs['raytace_no_optimization'].run()
		# self.programs['raytace_global_aabb_optimization'].run()
		# self.programs['raytace_bvh_optimization_scrapped'].run()
		# self.programs['raytace_bvh_optimization'].run()
		self.programs['raytace_bvh_optimization_fixed'].run()


if __name__ == "__main__":
	App().run()