#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

enum inter_device_comm{
	VR,	// Vertex Refinement method with both offline and online methods.
	VRONLINE,	// Vertex refinement with online-only method.
	MS,	// Maximal Set method.
	ALL	// Basically no method. Transfer all the vertices.
};

// Enforce warp size to be 32 at compile time.
static const unsigned int WARP_SIZE = 32;
static const unsigned int WARP_SIZE_SHIFT = 5;

#endif /* GLOBALS_HPP_ */
