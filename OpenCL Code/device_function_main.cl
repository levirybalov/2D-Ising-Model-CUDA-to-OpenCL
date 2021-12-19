#define FLAG_PRINT_SPINS 0
// define FLAG_ENERGY 1 to calculate energy, FLAG_ENERGY 0 to calculate magnetization
#define FLAG_ENERGY 1
#define T_START 3.00
#define T_FACTOR 0.999
#define T_END 2.00
#define GLOBAL_ITERATIONS 100
#define RANDOM_A 1664525
#define RANDOM_B 1013904223

#define BLOCK_SIZE 256

// n = one side of the lattice; N = number of lattice sites
const unsigned int N = 4 * BLOCK_SIZE * BLOCK_SIZE;
const unsigned int n = 2 * BLOCK_SIZE;
/****
 *
 *  Device function main
 *
 */
//__kernel void device_function_main(__global int* S, __global int* out, __global int* R, float t, bool flag) {
__kernel void device_function_main(__global int* S, __global int* out, __global int* R, float t, int flag) {

	//Energy variable
	// there is a dH for every thread in every block
	int dH = 0;
	float exp_dH_4 = exp(-(4.0) / t);
	float exp_dH_8 = exp(-(8.0) / t);

	//Allocate shared memory
	// shared memmory is allocated per thread block
	__local int r[BLOCK_SIZE];

	//Load random data
	// recall, as just mentioned above, r is allocated per thread block
	r[get_local_id(0)] = R[get_local_id(0) + BLOCK_SIZE * get_group_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);
	// first part of first conditional -> will hit if flag == true; executes steps (a) and (b) described in the paper
	if (flag) {

		// the top left site of each 2x2 cell are dealt with in the following order:
		// 1) top left cell
		// 2) top cells that aren't the leftmost
		// 3) leftmost cells that aren't the top
		// 4) all other cells

		//Create new random numbers
		r[get_local_id(0)] = RANDOM_A * r[get_local_id(0)] + RANDOM_B;

		//Spin update top left
		if (get_group_id(0) == 0) { //Top
			if (get_local_id(0) == 0) { //Left
			  // so here we are just accessing the top left site
				dH = 2 * S[2 * get_local_id(0)] * (
					S[2 * get_local_id(0) + 1] + // site to the right
					S[2 * get_local_id(0) - 1 + 2 * BLOCK_SIZE] + // site to the "left" (wrapping around)
					S[2 * get_local_id(0) + 2 * BLOCK_SIZE] + // site below; notice that these are linear indices
												   // so this site and the previous one only differ by 1
					S[2 * get_local_id(0) + N - 2 * BLOCK_SIZE]); // site "above" (wrapping around)
			}
			// top row minus leftmost cell
			else {
				dH = 2 * S[2 * get_local_id(0)] * (
					S[2 * get_local_id(0) + 1] + // site to the left
					S[2 * get_local_id(0) - 1] + // site to the right
					S[2 * get_local_id(0) + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + N - 2 * BLOCK_SIZE]); // site "above" (wrapping around)
			}
		}
		else { // not top
			if (get_local_id(0) == 0) { //Left
			  // index below implies that we are accessing sites 4*BLOCK_SIZE, 8*BLOCK_SIZE, 12*BLOCK_SIZE, ...
			  // so leftmost column, but not top point
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) - 1 + 2 * BLOCK_SIZE] + // site to the "left" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) - 2 * BLOCK_SIZE]); // site above
			}
			else {
				// all other cells
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) - 1] + // site to the left
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) - 2 * BLOCK_SIZE]); // site above
			}
		}

		// if Hamiltonian == 4
		if (dH == 4) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_4) {
				S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)];
			}
		}
		// if Hamiltonian == 8
		else if (dH == 8) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_8) {
				S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)];
			}
		}
		// if Hamiltonian <= 0
		else {
			S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)];
		}

		//Create new random numbers
		r[get_local_id(0)] = RANDOM_A * r[get_local_id(0)] + RANDOM_B;

		//Spin update bottom right
		// notice that indices are offest by a constant of 2*BLOCK_SIZE;
		// first linear index here will be 513, so bottom right of top left square, as expected
		if (get_group_id(0) == BLOCK_SIZE - 1) { //Bottom
			if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2] + // site to the "right" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] +  // site to the left
					S[2 * get_local_id(0) + 1] + // site "below" (wrapping around)
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)]); // site above (2*BLOCK_SIZE removed)
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 2] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site to the left
					S[2 * get_local_id(0) + 1] + // site "below" (wrapping around)
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)]); // site above (2*BLOCK_SIZE removed)
			}
		}
		else {
			if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2] + // site to the "right" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site to the left
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * (get_group_id(0) + 1)] + // site below (2*BLOCK_SIZE added)
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)]); // site above (2*BLOCK_SIZE removed)
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 2] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site to the left
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * (get_group_id(0) + 1)] + // site below (2*BLOCK_SIZE added)
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)]); // site above (2*BLOCK_SIZE removed)
			}
		}

		if (dH == 4) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_4) {
				S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
			}
		}
		else if (dH == 8) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_8) {
				S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
			}
		}
		else {
			S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	// second part of first conditional -> will hit if flag == false; will execute steps (c) and (d)
	else {

		//Create new random numbers
		r[get_local_id(0)] = RANDOM_A * r[get_local_id(0)] + RANDOM_B;

		//Spin update top right
		if (get_group_id(0) == 0) { //Top
			if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				dH = 2 * S[2 * get_local_id(0) + 1] * (
					S[2 * get_local_id(0) + 2 - 2 * BLOCK_SIZE] + // site to the "right". wrapping around
					S[2 * get_local_id(0)] + // site to the left
					S[2 * get_local_id(0) + 1 + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 1 + N - 2 * BLOCK_SIZE]); // site "above", wrapping around
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 1] * (
					S[2 * get_local_id(0) + 2] + // site to the right
					S[2 * get_local_id(0)] + // site to the left
					S[2 * get_local_id(0) + 1 + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 1 + N - 2 * BLOCK_SIZE]); // site "above", wrapping around
			}
		}
		else {
			if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 - 2 * BLOCK_SIZE] + // site to the "right", wrapping around
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] + // site to the left
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) - 2 * BLOCK_SIZE]); // site above
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] + // site to the left
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] + // site below
					S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) - 2 * BLOCK_SIZE]); //site above
			}
		}

		if (dH == 4) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_4) {
				S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)];
			}
		}
		else if (dH == 8) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_8) {
				S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)];
			}
		}
		else {
			S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] = -S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)];
		}

		//Create new random numbers
		r[get_local_id(0)] = RANDOM_A * r[get_local_id(0)] + RANDOM_B;

		//Spin update bottom left
		if (get_group_id(0) == BLOCK_SIZE - 1) { //Bottom
			if (get_local_id(0) == 0) { //Left
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1) - 1] + // site to the "left" (wrapping around)
					S[2 * get_local_id(0)] + // site "below" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)]); // site above
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE - 1] + // site to the left
					S[2 * get_local_id(0)] + // site "below" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)]); // site above
			}
		}
		else {
			if (get_local_id(0) == 0) { //Left
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1) - 1] + // site to the "left" (wrapping around)
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1)] + // site below
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)]); // site above
			}
			else {
				dH = 2 * S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE + 1] + // site to the right
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE - 1] + // site to the left
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1)] + // site below
					S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)]); // site above
			}
		}

		if (dH == 4) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_4) {
				S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
			}
		}
		else if (dH == 8) {
			if (fabs(r[get_local_id(0)] * 4.656612e-10) < exp_dH_8) {
				S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
			}
		}
		else {
			S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
		}

	}

	//Transfer random data back to global memory
	// this is saving the current set of random numbers, so they are not reused
	// (they will be loaded and modified when this GPU kernel is called again)
	R[get_local_id(0) + BLOCK_SIZE * get_group_id(0)] = r[get_local_id(0)];

	if (!flag) { // will hit if flag == false, i.e. after second set of spin updates

	  //For reduction shared memory array r is used
		if (FLAG_ENERGY) {

			//Calc energy
			// recall that each 2x2 cell is represented by a single thread, which has its own dH
			if (get_group_id(0) == BLOCK_SIZE - 1) { //Bottom
				if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				   // bottom right cell (last thread in last block):
				   // top left site * (site to the right + site below)
					dH = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// top right site * (site to the "right" (wrapping around) + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 1 - 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// bottom left site * (site to the right + site "below" (wrapping around))
						- S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0)])
						// bottom right site * (site to the "right" (wrapping around) + site "below" (wrapping around))
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2] + S[2 * get_local_id(0) + 1]);
				}
				else {
					// bottom block minus bottom right cell:
					// top left site * (site to the right + site below)
					dH = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// top right site * (site to the right + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// bottom left site * (site to the right + site "below" (wrapping around))
						- S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0)])
						// bottom right site * (site to the right + site "below" (wrapping around))
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 1]);
				}
			}
			else {
				if (get_local_id(0) == BLOCK_SIZE - 1) { //Right
				   // rightmost thread minus bottom right cell
				   // top left site * (site to the right + site below)
					dH = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// top right site * (site to the "right" (wrapping around) + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 1 - 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// bottom left site * (site to the right + site below)
						- S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1)])
						// bottom right site * (site to the "right" (wrapping around) + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * (get_group_id(0) + 1)]);
				}
				else {
					// all threads minus bottom block and rightmost threads
					// top left site * (site to the right + site below)
					dH = -S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// top right site * (site to the right + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)] * (S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 1] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE])
						// bottom left site * (site to the right + site below)
						- S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 1 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 4 * BLOCK_SIZE * (get_group_id(0) + 1)])
						// bottom right site * (site to the right + site below)
						- S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE] * (S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 + 2 * BLOCK_SIZE] + S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * (get_group_id(0) + 1)]);
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		else {

			//Calc magnetisation
			dH = S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0)]
				+ S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0)]
				+ S[2 * get_local_id(0) + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE]
				+ S[2 * get_local_id(0) + 1 + 4 * BLOCK_SIZE * get_group_id(0) + 2 * BLOCK_SIZE];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		//Save partial results back to shared memory in new structure
		// recall there is a dH for every thread, and an r for every block
		// and that dH is the reduced energy for each 2x2 cell
		r[get_local_id(0)] = dH;

		//Reduction on GPU
		// "A binary tree structure realizes a fast reduction of the partial values within a block. These partial results of
		// each block are stored at block-dependent positions in global memory..."
		for (unsigned int dx = 1;dx < BLOCK_SIZE;dx *= 2) {
			if (get_local_id(0) % (2 * dx) == 0) {
				r[get_local_id(0)] += r[get_local_id(0) + dx];
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		//Save in out
		// partial results from above loop are stored in r[0] for each block
		if (get_local_id(0) == 0) out[get_group_id(0)] = r[0];
	}
}