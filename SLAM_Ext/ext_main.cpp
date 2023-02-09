#include <iostream>
//#include <mpi.h>

#include "Cell.hpp"

#define TARGET_INPUT "geo.txt"

int main()
{
	

	Cell c(TARGET_INPUT);
	c.ShowBasicCellInfo();

	// Calculate Core / Core-Shell Contribution
	c.CalcCoulombEnergy();
	c.CalcCoulombDerivative();
	// Calculate LonePair Contribution
	c.CalcLonePairCoulombEnergy();		
	c.CalcLonePairCoulombDerivative();

	// This has to be called after StrainDerivatives are ready
	c.CalcLatticeDerivative();
	c.ShowEnergyDerivative();
	// progressing ... calculate lone pair derivatives
	

	
	//std::cout << "Compile Test" << std::endl;

	c.Finalise();
	return 0;
}
