import cupy as cp
import numpy as np
import time
from cupy import prof

size = 1<<24
h_answer = np.array(size)
c_answer = cp.array(size)

def warm_up():
	h_a = np.ones(size)
	h_b = np.ones(size)

	d_a = cp.asarray(h_a)
	d_b = cp.asarray(h_b)

	for _ in range (10):
		cp.dot(d_a, d_b)

# Basic dot product on the host
@cp.prof.TimeRangeDecorator()
def test_baseline():

	h_a = np.ones(size, np.int)
	h_b = np.ones(size, np.int)

	h_out = np.dot(h_a, h_b)
	np.testing.assert_equal(h_out, h_answer)

# Dot product on the host
# Notice in Systems there are no CUDA API calls in test_1
# That is because no device pointers were past to cp.dot
@cp.prof.TimeRangeDecorator()
def test_1():

	h_a = np.ones(size, np.int)
	h_b = np.ones(size, np.int)

	d_out = cp.dot(h_a, h_b)
	np.testing.assert_equal(d_out, h_answer)

# This example provides multiple ways to marker up your output with NVTX
# You will also see CUDA API calls
# 1. Notice in Systems there is a cudaMemcpy of 8 bytes
# fromt the device to the host
# 2. Notice that when the h_a and h_b are copied to the device
# they are pinned
# 3. Notice there is a cudaMalloc during Compute
# This is allocation for our output d_out
# 4. I'm not sure why there is dead time after the assert
# 5. print(d_out.device) tells me which GPU the data resides on.
@cp.prof.TimeRangeDecorator("Test 2", 2)
def test_2():

	cp.cuda.nvtx.RangePush("Init", 3)
	h_a = np.ones(size, np.int)
	h_b = np.ones(size, np.int)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Upload", 4):
		d_a = cp.asarray(h_a)
		d_b = cp.asarray(h_b)

	with cp.prof.time_range("Dot", 5):
		d_out = cp.dot(d_a, d_b)

	print(d_out.device)

	with cp.prof.time_range("Assert", 6):
		np.testing.assert_equal(d_out, c_answer)

# In this example we allocate and initialize on the GPU
# 1. We preallocate d_out on the GPU and pass in as argument
# 2. Compare using the np.testing.assert_equal from test_2
# to cp.testing.assert_array_equal. There is less communication between
# the host and device when compare device arrays
# 3. Notice this is no cudaMalloc(s) for d_a and d_b. It looks like cupy
# is using old allocations
@cp.prof.TimeRangeDecorator()
def test_3a():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Dot", 5):
		cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Assert", 6):
		cp.testing.assert_array_equal(d_out, c_answer)

# In this example we increase the size of d_a and d_b and 
# see it cause the cudaMalloc to be executed again
# If we change * to // there is no cudaMalloc
@cp.prof.TimeRangeDecorator()
def test_3b():

	scalar = 2
	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size*scalar, cp.int)
	d_b = cp.ones(size*scalar, cp.int)
	d_out = cp.ones(1, cp.int)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Dot", 5):
		cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Assert", 6):
		cp.testing.assert_array_equal(d_out, c_answer*scalar)

# In this example we pass the output from cp.dot into
# cp. multiply
# 1. Notice there is no cudaMalloc for d_out2. The data is 
@cp.prof.TimeRangeDecorator()
def test_4a():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Dot", 5):
		cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Multiply", 7):
		d_out2 = cp.multiply(d_out, 2)

	with cp.prof.time_range("Assert", 6):
		cp.testing.assert_array_equal(d_out2, c_answer * 2)

# Same as test4a, but with preallocation and passing in output
# array
@cp.prof.TimeRangeDecorator()
def test_4b():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.ones(1, cp.int)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Dot", 5):
		cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Multiply", 7):
		cp.multiply(d_out, 2, d_out2)

	with cp.prof.time_range("Assert", 6):
		cp.testing.assert_array_equal(d_out2, c_answer * 2)

# In this example we create and use a non-default stream
# 1. Notice in the left-hand column on System you now see
# Stream X, where X is an arbitrary number.
# 2. You don't see a huge overlap because the multiply kernel
# is most likely utilizing all the compute resourse. That being 
# said, you see an overlap when the 2nd kernel is launched.
# This is overlap on the tail-effect of the 1st kernel
@cp.prof.TimeRangeDecorator()
def test_5a():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.empty_like(d_a)
	cp.cuda.nvtx.RangePop()

	stream_1 = cp.cuda.stream.Stream()
	stream_2 = cp.cuda.stream.Stream()

	with cp.prof.time_range("Dot", 5):
		with stream_1:
			cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Multiply", 7):
		with stream_1:
			cp.multiply(d_a, d_b, d_out2)
	
	with cp.prof.time_range("Multiply_New_Stream", 7):
		with stream_2:
			cp.multiply(d_a, d_b, d_out2)

# This examples is the same as 5a, with some redundant work
# If you run is a few times, you should see an overlap between the streams
@cp.prof.TimeRangeDecorator()
def test_5b():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.empty_like(d_a)
	cp.cuda.nvtx.RangePop()

	stream_1 = cp.cuda.stream.Stream()
	stream_2 = cp.cuda.stream.Stream()

	with cp.prof.time_range("Dot", 5):
		with stream_1:
			cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("Multiply", 7):
		with stream_1:
			cp.multiply(d_a, d_b, d_out2)

	with cp.prof.time_range("Combo", 7):
		with stream_1:
			cp.dot(d_a, d_b, d_out)
			cp.multiply(d_a, d_b, d_out2)
	
	with cp.prof.time_range("Multiply_New_Stream", 7):
		with stream_2:
			cp.multiply(d_a, d_b, d_out2)

# In this example, just running cp.multiply, copy the result
# back to the host. Then launching cp.dot. 
# 1. Notice all calls are in the default stream
# 2. cp.dot must wait until d_out2.get() is finished
@cp.prof.TimeRangeDecorator()
def test_6a():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.empty_like(d_a)
	cp.cuda.nvtx.RangePop()

	with cp.prof.time_range("Multiply", 7):
		cp.multiply(d_a, d_b, d_out2)
	
	h_out2 = d_out2.get()

	with cp.prof.time_range("Dot", 5):
		cp.dot(d_a, d_b, d_out)

	h_out = d_out.get()

# In this example we move cp.multiply and cp.dot into
# separate streams. 
# 1. Notice cp.dot still executes after d_out2.get() because
# d_out2.get() is still in the default stream
@cp.prof.TimeRangeDecorator()
def test_6b():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.empty_like(d_a)
	cp.cuda.nvtx.RangePop()

	stream_1 = cp.cuda.stream.Stream()
	stream_2 = cp.cuda.stream.Stream()

	with cp.prof.time_range("Multiply", 7):
		with stream_1:
			cp.multiply(d_a, d_b, d_out2)
	
	h_out2 = d_out2.get()

	with cp.prof.time_range("Dot", 5):
		with stream_2:
			cp.dot(d_a, d_b, d_out)

	h_out = d_out.get()

# In this example we move both .get() to separate streams,
# but cp.dot is still running after the d_out2.get...
# This is because we didn't pinned d_out2. 
# 1. Notice that we are now using cudaMemcpyAsync, but 
# the copy itself is on pageable memory
@cp.prof.TimeRangeDecorator()
def test_6c():

	cp.cuda.nvtx.RangePush("Init", 3)
	d_a = cp.ones(size, cp.int)
	d_b = cp.ones(size, cp.int)
	d_out = cp.ones(1, cp.int)
	d_out2 = cp.empty_like(d_a)
	cp.cuda.nvtx.RangePop()

	stream_1 = cp.cuda.stream.Stream()
	stream_2 = cp.cuda.stream.Stream()

	with cp.prof.time_range("Multiply", 7):
		with stream_1:
			cp.multiply(d_a, d_b, d_out2)
	
	h_out2 = d_out2.get(stream_1)

	with cp.prof.time_range("Dot", 5):
		with stream_2:
			cp.dot(d_a, d_b, d_out)

	h_out = d_out.get(stream_2)

# This example pins memory using a pinned memory
# 1. Notice the overlap with cp.doc and d_out2.get()
# 2. Notice cp.ones are in a separate stream. We must synchronize
# because output is required for cp.multiply
# 3. h_out and h_out2 are added to pinned memory pool
# 4. Notice in Systems cudaFreeHost is called automatically
# 5. It takes a lot of time to pin memory. Better to do at startup
@cp.prof.TimeRangeDecorator()
def test_6d():

	pinned_memory_pool = cp.cuda.PinnedMemoryPool()
	cp.cuda.set_pinned_memory_allocator(pinned_memory_pool.malloc)

	def _pin_memory(array):
		mem = cp.cuda.alloc_pinned_memory(array.nbytes)
		ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
		ret[...] = array
		return ret

	h_out = np.empty(1, np.int)
	h_out2 = np.empty(size, np.int)

	with cp.prof.time_range("Pin", 3):
		x_pinned_h_out = _pin_memory(h_out)
		x_pinned_h_out2 = _pin_memory(h_out2)
	
	stream_fill = cp.cuda.stream.Stream()
	stream_1 = cp.cuda.stream.Stream()
	stream_2 = cp.cuda.stream.Stream()

	with cp.prof.time_range("Init", 3):
		with stream_fill:
			d_a = cp.ones(size, cp.int)
			d_b = cp.ones(size, cp.int)

	d_out = cp.empty(1, cp.int)
	d_out2 = cp.empty(size, cp.int)

	stream_fill.synchronize()

	with cp.prof.time_range("Multiply", 7):
		with stream_1:
			cp.multiply(d_a, d_b, d_out2)

	with cp.prof.time_range("d_out2.get", 7):
		d_out2.get(stream_1, out=x_pinned_h_out2)

	with cp.prof.time_range("Dot", 5):
		with stream_2:
			cp.dot(d_a, d_b, d_out)

	with cp.prof.time_range("d_out.get", 7):
		d_out.get(stream_2, out=x_pinned_h_out)


if __name__ == "__main__":

	warm_up(); time.sleep(0.1)
	test_baseline(); time.sleep(0.1)
	test_1(); time.sleep(0.1)
	test_2(); time.sleep(0.1)
	test_3a(); time.sleep(0.1)
	test_3b(); time.sleep(0.1)
	test_4a(); time.sleep(0.1)
	test_4b(); time.sleep(0.1)
	test_5a(); time.sleep(0.1)
	test_5b(); time.sleep(0.1)
	test_6a(); time.sleep(0.1)
	test_6b(); time.sleep(0.1)
	test_6c(); time.sleep(0.1)
	test_6d(); time.sleep(0.1)

