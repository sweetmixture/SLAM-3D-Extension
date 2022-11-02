#ifndef __LONEPAIR_MATRIX_H
#define __LONEPAIR_MATRIX_H

#define EV_UNIT (14.39964390675221758120)
//#define TO_BOHR_RADII (0.52917721067)							// div Angstrom -> Bohr, mul Bohr -> Angstrom
#define TO_BOHR_RADII (0.529177249)							// Wiki Base
#define HA_TO_EV_UNIT (EV_UNIT/TO_BOHR_RADII)						// mul Ha       -> eV
#define FHA_TO_FEV_UNIT (EV_UNIT/TO_BOHR_RADII/TO_BOHR_RADII)				// mul Ha/Bohr  -> eV/Angstrom
#define FFHA_TO_FFEV_UNIT (EV_UNIT/TO_BOHR_RADII/TO_BOHR_RADII/TO_BOHR_RADII)		// mul Ha/Bohr^2-> eV/Angstrom^2

//#include "Atom.hpp"

#include <Eigen/Core>
#include "Integral_lib.hpp"

#define GRID_W 2048		// Integral grid dense level 10^1

#define MIN(a,b)        ((a)>=(b)?(b):(a))

class LonePairMatrix
{
public:

	//N Integrate Variables

	double grid_weight;

	Eigen::Matrix4d transform_matrix;		// Raw 4x4 transformation matrix
	Eigen::Matrix3d transform_matrix_shorthand;	// Lower 3x3 block diag-matrix of the 4x4

	const int b_serach( const double dist, const std::vector<double>& integral_knot );

	const Eigen::Matrix4d& GetTransformationMatrix( const Eigen::Vector3d& Rij );	// Sets 'transform_matrix*'
	// Inverse Transformation - H_global = P_transpose * H_local * P
	// Direct  Transformation - H_local  = P * H_local * P_transpose
};

class LonePairMatrix_H : public LonePairMatrix	// 'public' specification is required ... take the public features as public
{

public: 

///	///	///	///	///	///	///	///
	void test()				// Binary Link Test
	{	printf("LonePairMatrix@@@@\n");
	}
	void test2();

	// Test Function for Validation
	const double NIntegral_test_real( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4] );
///	///	///	///	///	///	///	///

	// Real Space position integral ... in a general reference frame
	double real_position_integral( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4] );

	// Reci Space Self : (ss) / (xx=yy=zz)
	double reci_self_integral_ss( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig );
	double reci_self_integral_xx( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig );
	// Reci Space Self Derivatives : (sx=sy=sz) ... different way order same ... these functions in a general reference frame
	double reci_self_integral_sx_grad_x( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig );

	// Real Space Integral - LP...PointCharge(pc) Interaction
	double real_ss_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_sz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_zz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );

	// Test ... the grad is w.r.t pc (by its displacement of the above h_pc) Not!!! for the LP-Core ... therefore to get grad LP-Core, its sign must be inversed.
	double real_sx_grad_x_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xz_grad_x_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_ss_grad_z_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_sz_grad_z_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xx_grad_z_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_zz_grad_z_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );

	// Test .. the 2nd grad is w.r.t pc 
	double real_ss_grad2_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_sz_grad2_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xx_grad2_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_yy_grad2_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_zz_grad2_xx_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );

	double real_xy_grad2_xy_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );

	double real_sx_grad2_xz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xz_grad2_xz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );

	double real_ss_grad2_zz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_sz_grad2_zz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_xx_grad2_zz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	double real_zz_grad2_zz_pc( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double sig, const double d );
	// Real Space Integral - LP...LP Interaction


	// Reci Space Integral - LP COSINE Part
	double reci_ss_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	double reci_xx_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );		// xx = yy
	double reci_zz_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	// Reci Space Integral - LP   SINE Part
	double reci_sz_sin( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );

	// Reci Space Integral - LP COSINE Part derivative w.r.t. 'g' vector components
	double reci_xz_grad_gx_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	double reci_ss_grad_gz_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	double reci_xx_grad_gz_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	double reci_zz_grad_gz_cos( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	// Reci Space Integral - LP   SINE Part derivative w.r.t. 'g' vector components
	double reci_sx_grad_gx_sin( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	double reci_sz_grad_gz_sin( const std::vector<double>& integral_knot, const std::vector<double> (&Rs)[4], const std::vector<double> (&Rp)[4], const double g );
	


};

#endif




















