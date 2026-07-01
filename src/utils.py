"""
Helper Functions
"""
import os
import numpy as np
from engine import mgl


def data_to_flat_np_array(data: list, indices: list):
	data = [data[index] for triangle in indices for index in triangle]
	return np.array(data, dtype='f4')


def read_file(filedir:str, filename:str):
	with open(os.path.join(filedir, filename), "r") as rfs:
		content = rfs.read()

	return content

def normalize_vector(v:np.ndarray):
	"""
	Using the small constant to prevent the zero division error
	is the most robust, superseding even using the condition branching:
	if np.linalg.norm(v) == 0: return v
	"""
	return v / (np.linalg.norm(v) + 1e-12)

def calculate_normal(v0:np.ndarray, v1:np.ndarray):
	"""Calculates the Normal from Two 3D arrays representing 3D vectors"""
	return np.cross(normalize_vector(v0), normalize_vector(v1))

def set_uniform(shaderProgram, uname, value, couldThrow):
	try:
		if not uname in couldThrow:
			couldThrow[uname] = True
		shaderProgram[uname] = value
	except Exception as e:
		if couldThrow[uname]:
			print(f"Uniform Error for {uname}: {str(e)}")
			couldThrow[uname] = False

def set_complex_uniform(shaderProgram, uname, value, couldThrow):
	try:
		if not uname in couldThrow:
			couldThrow[uname] = True
		shaderProgram[uname].write(value)
	except Exception as e:
		if couldThrow[uname]:
			print(f"Uniform Error for {uname}: {str(e)}")
			couldThrow[uname] = False

def upload_flat_data_as_texture(ctx, data, dtype='i4'):
	"""
	ctx: Moderngl Context
	data: either np.ndarray or array.array (C-style)
	dtype: commonly either 'i4' or 'f4''
 
	Use 1D textures when the data's size is less than 8k-16k.
	Use 2D textures for much larger meshes
	This is because 1D textures' width N works up to the GPU limit `GPU_MAX_TEXTURE_SIZE`
	which is often >= 16384.

	If N (count of data) is larger than this, use a 2D textures which
	pack indices into a 2D array of size widthxheight >= N.
	E.g. ~10M indices/data count  use 4096 x 2500 texture.
	Modern GPUs often support 16384^2 2D textures, which is sufficient
	"""
	# Create a 1D texture (width=N, height=1) with signed 32-bit integers
	tex1d = ctx.texture((len(data), 1), components=1, dtype=dtype)
	tex1d.repeat_x = False  # No Wrap
	tex1d.filter =  (mgl.NEAREST, mgl.NEAREST)
	tex1d.write(data.tobytes()) # upload


def upload_flat_data_as_ssbo(ctx, bindindex: int, data:np.ndarray, dtype='f4', buffertype="int"):
	"""
	Uploads array data as Shader Storage Buffer Object

	ctx: the live moderngl context	
	bindindex: each ssbo has an integer index to differentiate them.
	data: the array data to be uploaded to GPU
	dtype: the type in which the array data should be interpreted as
	buffertype: either 'int', 'float', 'vec2', 'vec3'
	"""
	match buffertype:
		case 'int' | 'float':
			# data: N ints (flat list)
			data = np.array(data, dtype=dtype)
			buffer = ctx.buffer(data.tobytes())
			buffer.bind_to_storage_buffer(bindindex)
		case 'vec2':
			data = np.array(data, dtype=dtype).reshape(-1, 2)
			buffer = ctx.buffer(data.tobytes())
			buffer.bind_to_storage_buffer(bindindex)
		case 'vec3' | 'vec4':
			data = np.array(data, dtype=dtype).reshape(-1, 4)
			buffer = ctx.buffer(data.tobytes())
			buffer.bind_to_storage_buffer(bindindex)