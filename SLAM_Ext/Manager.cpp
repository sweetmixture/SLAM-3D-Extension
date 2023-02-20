#include "Manager.hpp"


// Debugging Defines
#define DERIVATIVE_CHECK
//#define SHOW_LP_MATRIX
//#define LPLP_CHECK
//#define SCF_LOG_DEBUG


// Essential Defines ...
#define IS_NAN
#define BOOST_MEMO	// INTEGRAL BOOST - SCF - Memoization scheme

int kDelta( const int u, const int v )
{	if( u == v ){ return 1; }
	else{ return 0; }
}

void ShowMatrix( const Eigen::Matrix4d& m )
{
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	printf("%20.12e\t",m(i,j));
		}
		std::cout << std::endl;
	}
	return;
}

void ShowVector( const Eigen::Vector3d v )
{
	for(int i=0;i<3;i++)
	{	printf("%20.12e\t",v(i));
	}
	std::cout << std::endl;
}

// Implement RealSpace Integrators - Input ... LonePair* / Vector to a species / sigma / IndexLonePair* / IndexSpecies

const Eigen::Matrix4d& Manager::set_h_matrix_real_pc( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(R);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp;					// calculating local h_matrix WS
	h_tmp.setZero();					// initialise

	// evalulation block
	h_tmp(0,0) = this->man_lp_matrix_h.real_ss_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp(0,3) = this->man_lp_matrix_h.real_sz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp(3,0) = h_tmp(0,3);
	h_tmp(1,1) = this->man_lp_matrix_h.real_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp(2,2) = h_tmp(1,1);
	h_tmp(3,3) = this->man_lp_matrix_h.real_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());

	this->real_lp_h_pc[lp_i][pc_i] = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp * this->man_lp_matrix_h.transform_matrix;	// inverse transformation

	return this->real_lp_h_pc[lp_i][pc_i];
}

void Manager::set_h_matrix_real_pc_derivative( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(R);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp_x, h_tmp_y, h_tmp_z;		// calculating local h_matrix WS
	Eigen::Matrix4d h_tmp_x_ws,h_tmp_y_ws,h_tmp_z_ws;
	Eigen::Vector3d v_loc, v_glo;

	h_tmp_x.setZero();					// initialise
	h_tmp_y.setZero();					// initialise
	h_tmp_z.setZero();					// initialise
	h_tmp_x_ws.setZero();					// initialise
	h_tmp_y_ws.setZero();					// initialise
	h_tmp_z_ws.setZero();					// initialise
	
	// 1. Compute first derivative integrals in a local symmetry
	h_tmp_x(0,1) = this->man_lp_matrix_h.real_sx_grad_x_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp_x(1,0) = h_tmp_x(0,1);
	h_tmp_x(1,3) = this->man_lp_matrix_h.real_xz_grad_x_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp_x(3,1) = h_tmp_x(1,3);

	h_tmp_y(0,2) = h_tmp_y(2,0) = h_tmp_x(0,1);	// y-sy = x-sx
	h_tmp_y(2,3) = h_tmp_y(3,2) = h_tmp_x(1,3);	// y-yz = x-sz
	
	h_tmp_z(0,0) = this->man_lp_matrix_h.real_ss_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp_z(0,3) = this->man_lp_matrix_h.real_sz_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp_z(3,0) = h_tmp_z(0,3);
	h_tmp_z(1,1) = this->man_lp_matrix_h.real_xx_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());
	h_tmp_z(2,2) = h_tmp_z(1,1);
	h_tmp_z(3,3) = this->man_lp_matrix_h.real_zz_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());

	// 2. Using the local elements; compute equivalent elements (in the global) in the local reference frame
	// note : h_tmp_*_ws are in the local reference frame, their x'/y'/z' element (local) has to be inversed to x/y/z (global) 
	h_tmp_x_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_x * this->man_lp_matrix_h.transform_matrix;
	h_tmp_y_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_y * this->man_lp_matrix_h.transform_matrix;
	h_tmp_z_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_z * this->man_lp_matrix_h.transform_matrix;

	// 3. Transform back to the global reference frame
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	v_loc << h_tmp_x_ws(i,j), h_tmp_y_ws(i,j), h_tmp_z_ws(i,j);
			v_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * v_loc;
			this->real_lp_h_pc_x[lp_i][pc_i](i,j) =  v_glo(0);
			this->real_lp_h_pc_y[lp_i][pc_i](i,j) =  v_glo(1);
			this->real_lp_h_pc_z[lp_i][pc_i](i,j) =  v_glo(2);
		}
	}
}

void Manager::set_h_matrix_real_pc_derivative2( LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(R);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp_d2[3][3];
	Eigen::Matrix4d h_d2_ws[3][3];
	Eigen::Matrix3d m_loc,m_glo;

	for(int i=0;i<3;i++) { for(int j=0;j<3;j++) { h_tmp_d2[i][j].setZero(); h_d2_ws[i][j].setZero(); }}	// Initialise workspace

	// [0][0] xx h_matrix loc
	h_tmp_d2[0][0](0,0) = this->man_lp_matrix_h.real_ss_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][0] - XX, (0,0) ss
	h_tmp_d2[0][0](0,3) = this->man_lp_matrix_h.real_sz_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][0] - XX, (0,3) sz
	h_tmp_d2[0][0](3,0) = h_tmp_d2[0][0](0,3);
	h_tmp_d2[0][0](1,1) = this->man_lp_matrix_h.real_xx_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][0] - XX, (1,1) xx
	h_tmp_d2[0][0](2,2) = this->man_lp_matrix_h.real_yy_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][0] - XX, (2,2) yy
	h_tmp_d2[0][0](3,3) = this->man_lp_matrix_h.real_zz_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][0] - XX, (3,3) zz

	// [0][1] xy h_matrix loc
	h_tmp_d2[0][1](1,2) = this->man_lp_matrix_h.real_xy_grad2_xy_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][1] - XY, (1,2) xy
	h_tmp_d2[0][1](2,1) = h_tmp_d2[0][1](1,2);

	// [0][2] xz h_matrix loc
	h_tmp_d2[0][2](0,1) = this->man_lp_matrix_h.real_sx_grad2_xz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][2] - XZ, (0,1) sx
	h_tmp_d2[0][2](1,0) = h_tmp_d2[0][2](0,1);
	h_tmp_d2[0][2](1,3) = this->man_lp_matrix_h.real_xz_grad2_xz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [0][2] - XZ, (1,3) xz
	h_tmp_d2[0][2](3,1) = h_tmp_d2[0][2](1,3);

	// [2][2] zz h_matix loc
	h_tmp_d2[2][2](0,0) = this->man_lp_matrix_h.real_ss_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [2][2] - ZZ, (0,0) ss
	h_tmp_d2[2][2](0,3) = this->man_lp_matrix_h.real_sz_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [2][2] - ZZ, (0,3) sz
	h_tmp_d2[2][2](3,0) = h_tmp_d2[2][2](0,3);
	h_tmp_d2[2][2](1,1) = this->man_lp_matrix_h.real_xx_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [2][2] - ZZ, (0,3) xx
	h_tmp_d2[2][2](2,2) = h_tmp_d2[2][2](1,1);
	h_tmp_d2[2][2](3,3) = this->man_lp_matrix_h.real_zz_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig,R.norm());	// [2][2] - ZZ, (3,3) zz

	// Below here, done by the orbital symmetry
	// [1][0] yx - [0][1] xy
	h_tmp_d2[1][0] = h_tmp_d2[0][1];
	// [1][1] yy ~ [0][0] xx
	h_tmp_d2[1][1] = h_tmp_d2[0][0];
	h_tmp_d2[1][1](1,1) = h_tmp_d2[0][0](2,2);	// YY - xx = XX - yy
	h_tmp_d2[1][1](2,2) = h_tmp_d2[0][0](1,1);	// YY - yy = XX - xx
	// [1][2] yz ~ [0][2] xz
	h_tmp_d2[1][2](0,2) = h_tmp_d2[1][2](2,0) = h_tmp_d2[0][2](0,1);	// YZ - sy = XZ - sx
	h_tmp_d2[1][2](2,3) = h_tmp_d2[1][2](3,2) = h_tmp_d2[0][2](1,3);	// YZ - yz = XZ - xz
	// [2][0] zx - [0][2] xz
	h_tmp_d2[2][0] = h_tmp_d2[0][2];
	// [2][1] zy - [1][2] yz
	h_tmp_d2[2][1] = h_tmp_d2[1][2];
	//// End of Local element Set

	// 2. Using the local elements; compute equivalent elements (in the global) in the local reference frame
	for(int i=0;i<3;i++){ for(int j=0;j<3;j++){ h_d2_ws[i][j] = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_d2[i][j] * this->man_lp_matrix_h.transform_matrix; }}

	// 3. Transform back to the global reference frame - i,j refer to basis functions
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	
			m_loc << h_d2_ws[0][0](i,j), h_d2_ws[0][1](i,j), h_d2_ws[0][2](i,j),
				 h_d2_ws[1][0](i,j), h_d2_ws[1][1](i,j), h_d2_ws[1][2](i,j),
				 h_d2_ws[2][0](i,j), h_d2_ws[2][1](i,j), h_d2_ws[2][2](i,j);

			m_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * m_loc * this->man_lp_matrix_h.transform_matrix_shorthand;

			this->real_lp_h_lp_xx[lp_i][pc_i](i,j) = m_glo(0,0); this->real_lp_h_lp_xy[lp_i][pc_i](i,j) = m_glo(0,1); this->real_lp_h_lp_xz[lp_i][pc_i](i,j) = m_glo(0,2);
			this->real_lp_h_lp_yx[lp_i][pc_i](i,j) = m_glo(1,0); this->real_lp_h_lp_yy[lp_i][pc_i](i,j) = m_glo(1,1); this->real_lp_h_lp_yz[lp_i][pc_i](i,j) = m_glo(1,2);
			this->real_lp_h_lp_zx[lp_i][pc_i](i,j) = m_glo(2,0); this->real_lp_h_lp_zy[lp_i][pc_i](i,j) = m_glo(2,1); this->real_lp_h_lp_zz[lp_i][pc_i](i,j) = m_glo(2,2);
		}
	}
}

const Eigen::Matrix4d& Manager::set_h_matrix_reci_cos( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& G, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(G);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp;					// calculating local h_matrix WS
	h_tmp.setZero();					// initialise

	// evalulation block
	h_tmp(0,0) = this->man_lp_matrix_h.reci_ss_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp(1,1) = this->man_lp_matrix_h.reci_xx_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp(2,2) = h_tmp(1,1);
	h_tmp(3,3) = this->man_lp_matrix_h.reci_zz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());

	this->reci_lp_h_pc[lp_i][pc_i] = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp * this->man_lp_matrix_h.transform_matrix;	// inverse transformation

	return this->reci_lp_h_pc[lp_i][pc_i];
}


void Manager::set_h_matrix_reci_derivative_cos( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& G, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(G);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp_x, h_tmp_y, h_tmp_z;		// calculating local h_matrix WS
	Eigen::Matrix4d h_tmp_x_ws,h_tmp_y_ws,h_tmp_z_ws;
	Eigen::Vector3d v_loc, v_glo;

	h_tmp_x.setZero();					// initialise
	h_tmp_y.setZero();					// initialise
	h_tmp_z.setZero();					// initialise
	h_tmp_x_ws.setZero();					// initialise
	h_tmp_y_ws.setZero();					// initialise
	h_tmp_z_ws.setZero();					// initialise
	
	// 1. Compute first derivative integrals in a local symmetry
	h_tmp_x(1,3) = this->man_lp_matrix_h.reci_xz_grad_gx_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp_x(3,1) = h_tmp_x(1,3);

	h_tmp_y(2,3) = h_tmp_y(3,2) = h_tmp_x(1,3);	// y-yz = x-sz
	
	h_tmp_z(0,0) = this->man_lp_matrix_h.reci_ss_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp_z(1,1) = this->man_lp_matrix_h.reci_xx_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp_z(2,2) = h_tmp_z(1,1);
	h_tmp_z(3,3) = this->man_lp_matrix_h.reci_zz_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());

	// 2. Using the local elements; compute equivalent elements (in the global) in the local reference frame
	// note : h_tmp_*_ws are in the local reference frame, their x'/y'/z' element (local) has to be inversed to x/y/z (global) 
	h_tmp_x_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_x * this->man_lp_matrix_h.transform_matrix;
	h_tmp_y_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_y * this->man_lp_matrix_h.transform_matrix;
	h_tmp_z_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_z * this->man_lp_matrix_h.transform_matrix;

	// 3. Transform back to the global reference frame
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	v_loc << h_tmp_x_ws(i,j), h_tmp_y_ws(i,j), h_tmp_z_ws(i,j);
			v_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * v_loc;
			this->reci_lp_h_pc_x[lp_i][pc_i](i,j) =  v_glo(0);
			this->reci_lp_h_pc_y[lp_i][pc_i](i,j) =  v_glo(1);
			this->reci_lp_h_pc_z[lp_i][pc_i](i,j) =  v_glo(2);
		}
	}
}


const Eigen::Matrix4d& Manager::set_h_matrix_reci_sin( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& G, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(G);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp;					// calculating local h_matrix WS
	h_tmp.setZero();					// initialise

	// evalulation block
	h_tmp(0,3) = this->man_lp_matrix_h.reci_sz_sin(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp(3,0) = h_tmp(0,3);

	this->reci_lp_h_pc[lp_i][pc_i] = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp * this->man_lp_matrix_h.transform_matrix;	// inverse transformation

	return this->reci_lp_h_pc[lp_i][pc_i];
}


void Manager::set_h_matrix_reci_derivative_sin( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& G, const int lp_i, const int pc_i )
{
	this->man_lp_matrix_h.GetTransformationMatrix(G);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix4d h_tmp_x, h_tmp_y, h_tmp_z;		// calculating local h_matrix WS
	Eigen::Matrix4d h_tmp_x_ws,h_tmp_y_ws,h_tmp_z_ws;
	Eigen::Vector3d v_loc, v_glo;

	h_tmp_x.setZero();					// initialise
	h_tmp_y.setZero();					// initialise
	h_tmp_z.setZero();					// initialise
	h_tmp_x_ws.setZero();					// initialise
	h_tmp_y_ws.setZero();					// initialise
	h_tmp_z_ws.setZero();					// initialise
	
	// 1. Compute first derivative integrals in a local symmetry
	h_tmp_x(0,1) = this->man_lp_matrix_h.reci_sx_grad_gx_sin(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp_x(1,0) = h_tmp_x(0,1);

	h_tmp_y(0,2) = h_tmp_y(2,0) = h_tmp_x(0,1);	// y-sy = x-sx
	
	h_tmp_z(0,3) = this->man_lp_matrix_h.reci_sz_grad_gz_sin(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,G.norm());
	h_tmp_z(3,0) = h_tmp_z(0,3);

	// 2. Using the local elements; compute equivalent elements (in the global) in the local reference frame
	// note : h_tmp_*_ws are in the local reference frame, their x'/y'/z' element (local) has to be inversed to x/y/z (global) 
	h_tmp_x_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_x * this->man_lp_matrix_h.transform_matrix;
	h_tmp_y_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_y * this->man_lp_matrix_h.transform_matrix;
	h_tmp_z_ws = this->man_lp_matrix_h.transform_matrix.transpose() * h_tmp_z * this->man_lp_matrix_h.transform_matrix;

	// 3. Transform back to the global reference frame
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	v_loc << h_tmp_x_ws(i,j), h_tmp_y_ws(i,j), h_tmp_z_ws(i,j);
			v_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * v_loc;
			this->reci_lp_h_pc_x[lp_i][pc_i](i,j) =  v_glo(0);
			this->reci_lp_h_pc_y[lp_i][pc_i](i,j) =  v_glo(1);
			this->reci_lp_h_pc_z[lp_i][pc_i](i,j) =  v_glo(2);
		}
	}
}


void Manager::InitialiseEnergy( Cell& C )
{


#ifdef DERIVATIVE_CHECK
using std::cout;
using std::endl;
LonePair* lp = nullptr;
int lp_id;

// FDM CHECK VARS
double delta = 0.01;
//double sig   = 1.85;
double sig   = 1.99347;
double g = 0.05;

for(int i=0;i<C.NumberOfAtoms;i++)
{	//cout << "index : " << i+1 << " / type : " << C.AtomList[i]->type << endl;
	if( C.AtomList[i]->type == "lone" )
	{	lp_id = i;
		lp    = static_cast<LonePair*>(C.AtomList[i]);
		break;
	}
}
//auto begin = std::chrono::system_clock::now();
//set_h_matrix_real_pc(lp,v,sig,lp_id,0);
//set_h_matrix_real_pc_derivative(lp,v,sig,lp_id,0);
//set_h_matrix_real_pc_derivative2(lp,v,sig,lp_id,0);
//auto end   = std::chrono::system_clock::now() - begin;
//auto time_s= std::chrono::duration<double>(end).count();
//cout << "Elapsed time (s) : lp - lp interaction" << endl;
//cout << time_s << " (s)" << endl;

if( lp != nullptr )
{
double ss,xx,zz;
double x_xz;
double z_ss,z_xx,z_zz;

cout << "Real Space Check      -----------------------------------------------------------" << endl;
Eigen::Vector3d R;
//R << -3.2,0.52,4.3;
//R << -8.4, -8.4, 0.0;
//R << -4.2, -4.2, 0.0;
//R << -2.1, -2.1, 0.0;

//R << -3, 0, 0;		// O
//R << -6, 0, 0;		// Err ...
//R << 0, 0, 6;			// Err ...
//R << 0, 0, 8;			// Err ...	// Integrals ----> Discontinuous part
//R << 0, 0, 8;			// Err ...	// Integrals ----> Discontinuous part
//R << 0, 0, 11.87;		// Err ...	// Integrals ----> Discontinuous part
//R<< 0, 0, 5.844;		// Err occurs after distance goes over the radial function

// Boundary Test
//R << 0, 0, 11.87;		// Err ...	// Integrals ----> Discontinuous part



//R << -3.2,0.52,4.3;
//R << -2.2,0.52,1.3;
R << -1.4,0.52,1.3;
cout << "Testing Coordinate: ";
for(int i=0;i<3;i++){ printf("%12.6lf\t",R(i)); }
cout << endl;
cout << "Length : " << R.norm() << endl;

// -8.4 X

ShowVector(R);

cout << "H Onsite" << endl;
//this->set_h_matrix_real_pc(lp,
//const Eigen::Matrix4d& Manager::set_h_matrix_real_pc( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i );
//void Manager::set_h_matrix_real_pc_derivative( /* IN/RES OUT */ LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i )
//void Manager::set_h_matrix_real_pc_derivative2( LonePair* lp, const Eigen::Vector3d& R, const double sig, const int lp_i, const int pc_i )
Eigen::Matrix4d h; h.setZero();

h = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",h(i,j)); } printf("\n"); }	// print h_matrix onsite

Eigen::Matrix4d fx[2], fy[2], fz[2]; fx[0].setZero(); fx[1].setZero(); fy[0].setZero(); fy[1].setZero(); fz[0].setZero(); fz[1].setZero();
Eigen::Matrix4d fdm_x,fdm_y,fdm_z;
// x forward
R[0] = R[0] + delta;
fx[0] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[0] = R[0] - delta;
// x backward
R[0] = R[0] - delta;
fx[1] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[0] = R[0] + delta;
// y forward
R[1] = R[1] + delta;
fy[0] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[1] = R[1] - delta;
// y backward
R[1] = R[1] - delta;
fy[1] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[1] = R[1] + delta;
// z forward
R[2] = R[2] + delta;
fz[0] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[2] = R[2] - delta;
// z backward
R[2] = R[2] - delta;
fz[1] = this->set_h_matrix_real_pc(lp,R,sig,lp_id,0);
R[2] = R[2] + delta;

// fdm calc 1d
fdm_x = (fx[0]-fx[1])/delta/2.;
fdm_y = (fy[0]-fy[1])/delta/2.;
fdm_z = (fz[0]-fz[1])/delta/2.;
cout << "FDM Derivatives Real" << endl;
cout << "FDMx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",fdm_x(i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "FDMy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",fdm_y(i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "FDMz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",fdm_z(i,j)); } printf("\n"); }	// print h_matrix onsite

cout << endl;
cout << "Derivatives Real analy" << endl;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);	// get 1st analytical derivatives
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite

cout << endl;
cout << "ERROR .... (difference)" << std::endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",(fdm_x(i,j) - this->real_lp_h_pc_x[lp_id][0](i,j))); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",(fdm_y(i,j) - this->real_lp_h_pc_y[lp_id][0](i,j))); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",(fdm_z(i,j) - this->real_lp_h_pc_z[lp_id][0](i,j))); } printf("\n"); }	// print h_matrix onsite



cout << endl;
cout << "2nd derivative FDM" << endl;
R[0] += delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "X Front" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[0] -= delta;
R[0] -= delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "X Back" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[0] += delta;

R[1] += delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "Y Front" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[1] -= delta;
R[1] -= delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "Y Back" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[1] += delta;

R[2] += delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "Z Front" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[2] -= delta;
R[2] -= delta;
this->set_h_matrix_real_pc_derivative(lp,R,sig,lp_id,0);
cout << "Z Back" << cout << endl;
cout << "dx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_x[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_y[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_pc_z[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
R[2] += delta;


// Calculate Analy Second Real derivatives ..
cout << endl;
cout << endl;
this->set_h_matrix_real_pc_derivative2(lp,R,sig,lp_id,0);
cout << "Derivatives 2 Real analy" << endl;
cout << "dxx Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_xx[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dxy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_xy[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dxz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_xz[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dyy Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_yy[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dyz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_yz[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite
cout << "dzz Real" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%14.8lf\t",this->real_lp_h_lp_zz[lp_id][0](i,j)); } printf("\n"); }	// print h_matrix onsite


cout << endl;
cout << endl;
cout << "Reciprocal Space Check -----------------------------------------------------------" << endl;

cout << "G FDM CHECK" << endl;
double ss_f,xx_f,zz_f;
double ss_b,xx_b,zz_b;
g = 25.124;
delta = 0.0001;
cout << "FDM Z Forward" << endl;
g = g + delta;
ss_f = this->man_lp_matrix_h.reci_ss_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
xx_f = this->man_lp_matrix_h.reci_xx_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
zz_f = this->man_lp_matrix_h.reci_zz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
printf("%20.12e\t%20.12e\t%20.12e\n",ss_f,xx_f,zz_f);
g = g - delta;
cout << "FDM Z Backward" << endl;
g = g - delta;
ss_b = this->man_lp_matrix_h.reci_ss_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
xx_b = this->man_lp_matrix_h.reci_xx_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
zz_b = this->man_lp_matrix_h.reci_zz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
printf("%20.12e\t%20.12e\t%20.12e\n",ss_b,xx_b,zz_b);
g = g + delta;
cout << "FDM Onsite" << endl;
z_ss = this->man_lp_matrix_h.reci_ss_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
z_xx = this->man_lp_matrix_h.reci_xx_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
z_zz = this->man_lp_matrix_h.reci_zz_grad_gz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,g);
printf("%20.12e\t%20.12e\t%20.12e\n",z_ss,z_xx,z_zz);
printf("%20.12e\t%20.12e\t%20.12e\n",(ss_f-ss_b)/2./delta,(xx_f-xx_b)/2./delta,(zz_f-zz_b)/2./delta);

cout << endl;
cout << "Transformation Check" << endl;
Eigen::Vector3d G;
G << -1.14,0.24,0.3;
for(int i=0;i<3;i++){ printf("%12.8lf\t",G(i)); }
cout << "Length: " << G.norm() << endl;
cout << endl;
cout << "H Onsite" << endl;
this->set_h_matrix_reci_cos(lp,G,lp_id,0);
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc[lp_id][0](i,j)); } cout << endl;}
cout << "dH Onsite" << endl;
this->set_h_matrix_reci_derivative_cos(lp,G,lp_id,0);
cout << "d/dgx" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_x[lp_id][0](i,j)); } cout << endl;}
cout << "d/dgy" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_y[lp_id][0](i,j)); } cout << endl;}
cout << "d/dgz" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_z[lp_id][0](i,j)); } cout << endl;}

cout << "Initiate FDM G" << endl;
Eigen::Matrix4d gx,gy,gz;
gx.setZero();
gy.setZero();
gz.setZero();

// X
cout << "FDM gx" << endl;
G(0) = G(0) + delta;
gx = this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(0) = G(0) - delta;
G(0) = G(0) - delta;
gx = gx - this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(0) = G(0) + delta;
gx = gx/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gx(i,j)); } cout << endl;}
// Y
cout << "FDM gy" << endl;
G(1) = G(1) + delta;
gy = this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(1) = G(1) - delta;
G(1) = G(1) - delta;
gy = gy - this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(1) = G(1) + delta;
gy = gy/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gy(i,j)); } cout << endl;}
// Z
cout << "FDM gz" << endl;
G(2) = G(2) + delta;
gz = this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(2) = G(2) - delta;
G(2) = G(2) - delta;
gz = gz - this->set_h_matrix_reci_cos(lp,G,lp_id,0);
G(2) = G(2) + delta;
gz = gz/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gz(i,j)); } cout << endl;}



cout << endl;
cout << "PosIntegral test" << endl;
double posint;
posint = this->man_lp_matrix_h.real_position_integral(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function);
printf("PosIntegral : %20.12lf\n",posint);

double lp_self_ss;
double lp_self_xx;
double lp_self_grad;

sig = 1.3214;

lp_self_ss = this->man_lp_matrix_h.reci_self_integral_ss(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig);
lp_self_xx = this->man_lp_matrix_h.reci_self_integral_xx(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig);
lp_self_grad = this->man_lp_matrix_h.reci_self_integral_sx_grad_x(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig);

cout << "lp_self ss / xx / lp_self_grad " << endl;
printf("%18.12lf\t%18.12lf\t%18.12lf\n",lp_self_ss,lp_self_xx,lp_self_grad);

cout << "### Terminating G-Space Recipe Dev" << endl;

cout << endl;
cout << endl;
cout << endl;
cout << "Testing SinPart" << endl;

cout << "G FDM CHECK" << endl;
delta = 0.0001;
G << -3.14,4.44,-0.3;
sig = 2.4214;
cout << "G Vector on Test: ";
printf("%18.12lf\t%18.12lf\t%18.12lf\n",G(0),G(1),G(2));
cout << "GNorm :           " << G.norm() << endl;

cout << endl;
cout << "H Onsite" << endl;
this->set_h_matrix_reci_sin(lp,G,lp_id,0);
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc[lp_id][0](i,j)); } cout << endl;}
cout << "dH Onsite" << endl;
this->set_h_matrix_reci_derivative_sin(lp,G,lp_id,0);
cout << "d/dgx" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_x[lp_id][0](i,j)); } cout << endl;}
cout << "d/dgy" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_y[lp_id][0](i,j)); } cout << endl;}
cout << "d/dgz" << endl;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",this->reci_lp_h_pc_z[lp_id][0](i,j)); } cout << endl;}

cout << "Initiate FDM G" << endl;
//Eigen::Matrix4d gx,gy,gz;
gx.setZero();
gy.setZero();
gz.setZero();

// X
cout << "FDM gx" << endl;
G(0) = G(0) + delta;
gx = this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(0) = G(0) - delta;
G(0) = G(0) - delta;
gx = gx - this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(0) = G(0) + delta;
gx = gx/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gx(i,j)); } cout << endl;}
// Y
cout << "FDM gy" << endl;
G(1) = G(1) + delta;
gy = this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(1) = G(1) - delta;
G(1) = G(1) - delta;
gy = gy - this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(1) = G(1) + delta;
gy = gy/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gy(i,j)); } cout << endl;}
// Z
cout << "FDM gz" << endl;
G(2) = G(2) + delta;
gz = this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(2) = G(2) - delta;
G(2) = G(2) - delta;
gz = gz - this->set_h_matrix_reci_sin(lp,G,lp_id,0);
G(2) = G(2) + delta;
gz = gz/2./delta;
for(int i=0;i<4;i++){ for(int j=0;j<4;j++){ printf("%20.12e\t",gz(i,j)); } cout << endl;}

} // if( lp != nullptr )
#endif	//  #define DEV_G_SPACE

	// Method Actual...
	C.energy_real_sum_cnt = 0;
	C.energy_reci_sum_cnt = 0;
	C.mono_real_energy = C.mono_reci_energy = C.mono_reci_self_energy = C.mono_total_energy = 0.;	// CLASSICAL ENERGY
}

void Manager::InitialiseDerivative( Cell& C )
{	
	C.derivative_real_sum_cnt = 0;
	C.derivative_reci_sum_cnt = 0;
	for(int i=0;i<C.NumberOfAtoms;i++) {	C.AtomList[i]->InitialiseDerivative();	}		// Initialise Derivative Field
	C.lattice_sd.setZero();										// Initialise Strain Drivative Field
}

//	Optimise Periodic Summation Workload
void Manager::InitialisePeriodicSysParameter( Cell& C )		// Prepare Parameters - Periodic Summation
{
	int NumberOfObject = 0;

	for(int i=0;i<C.NumberOfAtoms;i++)
	{	NumberOfObject++;
		//if( C.AtomList[i]->type == "shel" ) { NumberOfObject++; }	// ++ count for NOA if shel exists
	}

	C.sigma = pow(C.weight*NumberOfObject*M_PI*M_PI*M_PI/C.volume/C.volume,-1./6.);
	C.rcut  = std::sqrt(-log(C.accuracy)*C.sigma*C.sigma);
	C.gcut  = 2./C.sigma*std::sqrt(-log(C.accuracy));

	C.h_max  = static_cast<int>(C.rcut / C.real_vector[0].norm());
	C.k_max  = static_cast<int>(C.rcut / C.real_vector[1].norm());
	C.l_max  = static_cast<int>(C.rcut / C.real_vector[2].norm());
	C.ih_max = static_cast<int>(C.gcut / C.reci_vector[0].norm());
	C.ik_max = static_cast<int>(C.gcut / C.reci_vector[1].norm());
	C.il_max = static_cast<int>(C.gcut / C.reci_vector[2].norm());
}


////	////	////	////	////	////	////	////	////	////	////	////	////

////	Coulomb Interaction ( Periodic Summation )

////	////	////	////	////	////	////	////	////	////	////	////	////

void Manager::CoulombMonoMonoReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	Eigen::Vector3d Rij;
        // TransVector = h*a + k*b + l*c
        // Rij         = Ai.r - Aj.r - TransVector;

        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {       
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;

		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
        }       

	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		// 2. Handling Core - Shel
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		// 2. Handling Shel - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		// 2. Handling Core - Shel
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		// 3. Handling Shel - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		// 4. Handling Shel - Shel
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
        }       
}       

void Manager::CoulombMonoMonoSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qic,Qjc,Qis,Qjs;
	Eigen::Vector3d Rij;
        // TransVector = h*a + k*b + l*c
        // Rij         = Ai.r - Aj.r;
	
        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {       
		Qic = C.AtomList[i]->charge;
		Qjc = C.AtomList[j]->charge;
		C.mono_reci_self_energy += -0.5*(Qic*Qjc)*2./C.sigma/sqrt(M_PI) * C.TO_EV;
        }

	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// 1. Handling Core - Core
		Qic = C.AtomList[i]->charge;
		Qis = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qjc = C.AtomList[j]->charge;
		Qjs = static_cast<Shell*>(C.AtomList[j])->shel_charge;

		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->cart;

		if ( Rij.norm() != 0. )	// if shell / core is seperated !
		{
			C.mono_reci_self_energy += -0.5*(Qic*Qjc + Qis*Qis)*2./C.sigma/sqrt(M_PI) * C.TO_EV;
			C.mono_reci_self_energy += -(Qic*Qjs)/Rij.norm()*erf(Rij.norm()/C.sigma)* C.TO_EV;
		}
		else
		{
			C.mono_reci_self_energy += -0.5*(Qic*Qjc + Qic*Qis + Qis*Qic + Qis*Qis)*2./C.sigma/sqrt(M_PI) * C.TO_EV;
		}
			
        }       
}       

void Manager::CoulombMonoMonoReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;
        
        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {	
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		// 2. Handling Core - Shel
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		// 3. Handling Shel - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// 1. Handling Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		// 2. Handling Core - Shel
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		// 3. Handling Shel - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		// 4. Handling Shel - Shel
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
		C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
        }       
} 

////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

////	Geometric (RAW) Derivatives

void Manager::CoulombDerivativeReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;

        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
		
		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
// r_norm, r_sqr were missed!!!!
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact*Rij;

        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;

		// Shel - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
		
		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
// r_norm, r_sqr were missed!!!!

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact*Rij;

		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;

		// Shell - Shell
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact*Rij;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact*Rij;
        }       
}

void Manager::CoulombDerivativeSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
	
	double intact;

	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// Self - Core/Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		if ( r_norm != 0. )	// Case of Shell-Core separation is not zero
		{
			intact = 2.0 * (-0.5*Qi*Qj*(2./C.sigma/sqrt(M_PI)*exp(-r_sqr/C.sigma/C.sigma)/r_sqr - erf(r_norm/C.sigma)/r_norm/r_sqr) * C.TO_EV);
			// shell - core , core - shell counting twice

			C.AtomList[i]->cart_gd += intact * Rij;
			static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact * Rij;
		}
	}
}       

void Manager::CoulombDerivativeReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector /* G */)
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;
	double intact;
        
        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {	
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
	
		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact * TransVector;

		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
	
		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// Shell - Shell
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
	
		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact * TransVector;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact * TransVector;
        }       
}       

////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

////	Strain Derivative

void Manager::StrainDerivativeReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;
        
        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {	
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {	
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Shell - Shell
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
}       

void Manager::StrainDerivativeSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;

        if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" )    // Handling Core - Core (i.e., charge charge interaction);
	{
		// Self - Core/Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		if ( r_norm != 0. )	// Case of Shell-Core separation is not zero
		{
			intact = 2.0 * (-0.5*Qi*Qj*(2./C.sigma/sqrt(M_PI)*exp(-r_sqr/C.sigma/C.sigma)/r_sqr - erf(r_norm/C.sigma)/r_norm/r_sqr) * C.TO_EV);
			// leading "2.0 *" shell - core , core - shell counting twice

			C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
									C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
															C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		}

	}
}

void Manager::StrainDerivativeReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;
	double intact[4];

        if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											                                C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																			                                                                    C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														                                            C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																									 C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "core" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "shel" ) 
        {
		// Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// Core - Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// Shell - Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
		
		// Shell - Shell
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
}



////	////	////	////	////	////	////

////	LonePair_Member_Functions

////	////	////	////	////	////	////

void Manager::InitialiseSCF( Cell& C )
{
	LonePair* lp = nullptr;

	for(int i=0;i<C.NumberOfAtoms;i++)
	{
		if( C.AtomList[i]->type == "lone" )	
		{
			lp = static_cast<LonePair*>(C.AtomList[i]);
			lp->lp_real_position_integral = this->man_lp_matrix_h.real_position_integral( lp->lp_r, lp->lp_r_s_function, lp->lp_r_p_function );	// set realspace position integrals ... <s|rx|x> = <s|ry|y> = <s|rz|z>

			// Setting Temporal h_matrix zeroes
			lp->lp_h_matrix_tmp.setZero();
			// Setting Onsite LonePair Model Parameter Lambda
			lp->lp_h_matrix_tmp(1,1) = lp->lp_lambda;
			lp->lp_h_matrix_tmp(2,2) = lp->lp_lambda;
			lp->lp_h_matrix_tmp(3,3) = lp->lp_lambda;
			// Get EigenValues / EigenVectors
			lp->GetEigenSystem();
			/*
				Basically, what it does, setting the temporal matrices 'lp_h_matrix_tmp' of LonePair instance with the lp_lambda parameter only, and find the eigensyste.

				Here, the ground state eigenvalue will be '0' consisting of a pure 's' state, and the rests are with eigenvalue of Lambda(positive) formed of px/py/pz states

				The memberfunction 'GetEigenSystem' will also set the groundstate, saved in a member variable, 'lp->lp_gs_index'
			*/
		}
	}

	this->man_scf_lp_eval.clear(); 
	this->man_scf_lp_real_energy.clear();
	this->man_scf_lp_reci_energy.clear();
	this->man_scf_lp_total_energy.clear();
}

void Manager::InitialiseLonePairCalculation_Energy( Cell& C , const bool is_first_scf )
{
	for(int j=0;j<C.NumberOfAtoms;j++)
	{	for(int k=0;k<C.NumberOfAtoms;k++)
		{
			#ifdef BOOST_MEMO
			if( is_first_scf == true )	// only needs to be calculated in the first SCF cycle
			#endif
			{
				LPC_H_Real[j][k][0].setZero();	// Interaction with cores + LP cores
				LPC_H_Real[j][k][1].setZero();	// Interaction with shels

				LPC_H_Reci[j][k][0].setZero();
				LPC_H_Reci[j][k][1].setZero();
			}
			// else ... keep the pre-calculated value

			LPLP_H_Real[j][k].setZero();	// Interaction of LP<--->LP Real
			LPLP_H_Reci[j][k].setZero();	// Interaction of LP<--->LP Reciprocal
	
			// For Geometric Derivative Calculation Later - Real
			LPLP_H_Real_Aux[j][k][0].setZero(); LPLP_H_Real_Aux[j][k][1].setZero(); LPLP_H_Real_Aux[j][k][2].setZero();
			// For Geometric Derivative Calculation Later - Reciporcal
			LPLP_H_Reci_Aux[j][k][0].setZero(); LPLP_H_Reci_Aux[j][k][1].setZero();	LPLP_H_Reci_Aux[j][k][2].setZero();
			LPLP_H_Reci_Aux[j][k][3].setZero(); LPLP_H_Reci_Aux[j][k][4].setZero();	LPLP_H_Reci_Aux[j][k][5].setZero();
			LPLP_H_Reci_Aux[j][k][6].setZero(); LPLP_H_Reci_Aux[j][k][7].setZero();	LPLP_H_Reci_Aux[j][k][8].setZero();
			LPLP_H_Reci_Aux[j][k][9].setZero();
		}
	}

	C.lp_eval_sum = C.lp_real_energy = C.lp_reci_energy = C.lp_total_energy = 0.;
} 

void Manager::GetLonePairGroundState( Cell& C )	// Including Matrix Diagonalisaion + SetGroundState Index
{
	// This Function Assumes "LonePair::Eigen::Matrix4d lp_h_matrix_tmp" is Ready to be diagonalised
	std::vector<double> v(4);	// SPACE FOR HOLDING EIGEN VALUES
	double lp_scf_sum = 0.;		// TEMOPORALILY HOLDING GROUND STATE ENERGY SUM
	LonePair* lp = nullptr;

#ifdef SHOW_LP_MATRIX
Eigen::Matrix4d LPC_Real, LPLP_Real;
Eigen::Matrix4d LPC_Reci, LPLP_Reci;
#endif
	for(int i=0;i<C.NumberOfAtoms;i++)
	{
		if( C.AtomList[i]->type == "lone" )
		{
			lp = static_cast<LonePair*>(C.AtomList[i]);
/*
std::cout << "---------------- BEFORE Diagonalisation LP label : " << i << std::endl;
printf("H matrix ... \n");
std::cout << static_cast<LonePair*>(C.AtomList[i])->lp_h_matrix << std::endl;
printf("EigenValues  ... \n");
std::cout << static_cast<LonePair*>(C.AtomList[i])->lp_eigensolver.eigenvalues() << std::endl;
printf("EigenVectors ... \n");
std::cout << static_cast<LonePair*>(C.AtomList[i])->lp_eigensolver.eigenvectors() << std::endl;
*/
			// Setting Temporal h_matrix zeroes
			lp->lp_h_matrix_tmp.setZero();
			// Setting Onsite LonePair Model Parameter Lambda
			lp->lp_h_matrix_tmp(1,1) = lp->lp_lambda;
			lp->lp_h_matrix_tmp(2,2) = lp->lp_lambda;
			lp->lp_h_matrix_tmp(3,3) = lp->lp_lambda;
#ifdef SHOW_LP_MATRIX
LPC_Real.setZero(); LPLP_Real.setZero();
LPC_Reci.setZero(); LPLP_Reci.setZero();
#endif
			// SET TMP matrix
			//for(int j=0;j<MX_C;j++)
			for(int j=0;j<C.NumberOfAtoms;j++)
			{
				lp->lp_h_matrix_tmp += LPC_H_Real[i][j][0];	// Contribution by Core
				lp->lp_h_matrix_tmp += LPC_H_Real[i][j][1];	// Contribution by Shell
				// ---------------------------------------- calculated only when they are in the first scf cycle
				lp->lp_h_matrix_tmp += LPLP_H_Real[i][j];	// Contribution by LP-Density

				lp->lp_h_matrix_tmp += LPC_H_Reci[i][j][0];
				lp->lp_h_matrix_tmp += LPC_H_Reci[i][j][1];
				// ---------------------------------------- calculated only when they are in the first scf cycle
				lp->lp_h_matrix_tmp += LPLP_H_Reci[i][j];
#ifdef SHOW_LP_MATRIX
LPC_Real += LPC_H_Real[i][j][0];
LPLP_Real+= LPLP_H_Real[i][j];
LPC_Reci += LPC_H_Reci[i][j][0];
LPLP_Reci+= LPLP_H_Reci[i][j];
#endif
			}
#ifdef SHOW_LP_MATRIX
std::cout << "*** Real LPC\n";
ShowMatrix(LPC_Real);
std::cout << "*** Real LPLP\n";
ShowMatrix(LPLP_Real);
std::cout << "*** Reci LPC\n";
ShowMatrix(LPC_Reci);
std::cout << "*** Reci LPLP\n";
ShowMatrix(LPLP_Reci);
std::cout << "*** lp_h_matrix_tmp" << std::endl;
ShowMatrix(static_cast<LonePair*>(C.AtomList[i])->lp_h_matrix_tmp);
#endif

			lp->lp_h_matrix = lp->lp_h_matrix_tmp;
			// Copy lp_h_matrix_tmp -> (into) lp_h_matrix ... dialgonalisation target
			lp->lp_eigensolver.compute(lp->lp_h_matrix,true);
			// Diagonalise LonePair H matrix, compute_evec='true'
		
			// Get LonePair GroundState Index
			v[0] = lp->lp_eigensolver.eigenvalues()(0).real();
			v[1] = lp->lp_eigensolver.eigenvalues()(1).real();
			v[2] = lp->lp_eigensolver.eigenvalues()(2).real();
			v[3] = lp->lp_eigensolver.eigenvalues()(3).real();
			
			lp->lp_gs_index = std::min_element(v.begin(),v.end()) - v.begin();
			
			lp_scf_sum += lp->lp_eigensolver.eigenvalues()(lp->lp_gs_index).real();

#ifdef SHOW_LP_INFO
std::cout << "---------------- AFTER Diagonalisation LP label : " << i << std::endl;
printf("H matrix ... \n");
ShowMatrix(static_cast<LonePair*>(C.AtomList[i])->lp_h_matrix);
printf("EigenValues  ... \n");
std::cout << static_cast<LonePair*>(C.AtomList[i])->lp_eigensolver.eigenvalues() << std::endl;
printf("EigenVectors ... \n");
std::cout << static_cast<LonePair*>(C.AtomList[i])->lp_eigensolver.eigenvectors() << std::endl;
#endif

		}
	}
	C.lp_eval_sum = lp_scf_sum;
	C.lp_total_energy = C.lp_eval_sum + C.lp_real_energy + C.lp_reci_energy;

	this->man_scf_lp_eval.push_back(C.lp_eval_sum);	// Logging CycSum .. i.e., eval_sum
	this->man_scf_lp_real_energy.push_back(C.lp_real_energy);
	this->man_scf_lp_reci_energy.push_back(C.lp_reci_energy);
	this->man_scf_lp_total_energy.push_back(C.lp_total_energy);


#ifdef SCF_LOG_DEBUG
printf("!! Accumulated scf evals / lp_real_energies\n");
for(int i=0;i<this->man_scf_lp_eval.size();i++)
{
	printf("%d \t %20.12e\t%20.12e\t%20.12e\n",i+1,this->man_scf_lp_eval[i],this->man_scf_lp_real_energy[i],this->man_scf_lp_reci_energy[i]);
}
#endif

}// function end;

bool Manager::IsSCFDone( const double tol )			// Check If SCF Converged
{
	if( this->man_scf_lp_eval.size() < 2 )	// i.e., IF THIS IS THE 'FIRST SCF CYCLE'
	{	return false;
	}
	else	// IF IS THE CYCLES AFTHER THE FIRST
	{	if( fabs(this->man_scf_lp_eval[this->man_scf_lp_eval.size()-1] - this->man_scf_lp_eval[this->man_scf_lp_eval.size()-2]) > tol ) { return false; } // IF THE RECENT ENERGY PAIR DIFFERENCE IS LESS THAN THE TOLERANCE
		else{ return true; }
	}
}

void Manager::PrintSCFProfile( Cell& C )
{	
	for(int i=0;i<this->man_scf_lp_eval.size();i++)
	{	
		printf("   SCF(%d)\t%14.9lf\t%14.9lf\t%14.9lf\t%14.9lf\t%14.9lf\n",i+1,this->man_scf_lp_total_energy[i] + C.mono_total_energy,this->man_scf_lp_total_energy[i],
									this->man_scf_lp_eval[i],this->man_scf_lp_real_energy[i],this->man_scf_lp_reci_energy[i]);
	}
}

////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

////	LonePair_Energy

/* Supportive Functions - RealSpace i!=j cases */

void Manager::support_h_matrix_real( const LonePair* lp, const double& sigma, const Eigen::Vector3d& Rij, /* workspace */ Eigen::Matrix4d& h_mat_ws, /* out */ Eigen::Matrix4d& h_mat_out )
{
	this->man_lp_matrix_h.GetTransformationMatrix(Rij);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	h_mat_ws.setZero();

	// Evaluation
	h_mat_ws(0,0) = this->man_lp_matrix_h.real_ss_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws(0,3) = this->man_lp_matrix_h.real_sz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws(3,0) = h_mat_ws(0,3);
	h_mat_ws(1,1) = this->man_lp_matrix_h.real_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws(2,2) = h_mat_ws(1,1);
	h_mat_ws(3,3) = this->man_lp_matrix_h.real_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());

	// Inverse Transformation
	h_mat_out = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws * this->man_lp_matrix_h.transform_matrix;
}

void Manager::support_h_matrix_real_derivative( const LonePair* lp, const double& sigma, const Eigen::Vector3d& Rij, /* workspace */ Eigen::Matrix4d (&h_mat_ws)[3], /* out */ Eigen::Matrix4d (&h_mat_out)[3] )
{
	this->man_lp_matrix_h.GetTransformationMatrix(Rij);	// get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Vector3d v_loc, v_glo;	// workspace

#ifdef ISNAN
for(int i=0;i<4;i++)
{	for(int j=0;j<4;j++)
	{	if( std::isnan( this->man_lp_matrix_h.transform_matrix(i,j) ) )
		{	printf("transform isnan %20.12e\n",Rij.norm()); 
			printf("%20.12e\t%20.12e\t%20.12e\n",Rij[0],Rij[1],Rij[2]); exit(1);
}}}
#endif

	h_mat_ws[0].setZero(); h_mat_ws[1].setZero(); h_mat_ws[2].setZero();	// 1st derivatives ... w.r.t. 'j' of Ri -> Rj // dx dy dz - workspace

	// 1. Compute first derivative integrals in a local symmetry
	h_mat_ws[0](0,1) = this->man_lp_matrix_h.real_sx_grad_x_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](1,0) = h_mat_ws[0](0,1);
	h_mat_ws[0](1,3) = this->man_lp_matrix_h.real_xz_grad_x_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](3,1) = h_mat_ws[0](1,3);

	h_mat_ws[1](0,2) = h_mat_ws[1](2,0) = h_mat_ws[0](0,1);	// y-sy = x-sx
	h_mat_ws[1](2,3) = h_mat_ws[1](3,2) = h_mat_ws[0](1,3);	// y-yz = x-xz

#ifdef ISNAN
for(int i=0;i<4;i++)
{	for(int j=0;j<4;j++)
	{	if( std::isnan( h_mat_ws[0](i,j) ) )
		{	printf("x isnan %d\t%d\n",i,j);	exit(1);
}}}
#endif
	h_mat_ws[2](0,0) = this->man_lp_matrix_h.real_ss_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[2](0,3) = this->man_lp_matrix_h.real_sz_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[2](3,0) = h_mat_ws[2](0,3);
	h_mat_ws[2](1,1) = this->man_lp_matrix_h.real_xx_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[2](2,2) = h_mat_ws[2](1,1);
	h_mat_ws[2](3,3) = this->man_lp_matrix_h.real_zz_grad_z_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());

#ifdef ISNAN
for(int i=0;i<4;i++)
{	for(int j=0;j<4;j++)
	{	if( std::isnan( h_mat_ws[2](i,j) ) )
		{	printf("z isnan %d\t%d\n",i,j); exit(1);
}}}
#endif
	// 2. Using the local elements; compute equivalent elements (in the global) in the local reference frame
	// note : h_tmp_*_ws are in the local reference frame, their x'/y'/z' element (local) has to be inversed to x/y/z (global) 
	h_mat_ws[0] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[0] * this->man_lp_matrix_h.transform_matrix;
	h_mat_ws[1] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[1] * this->man_lp_matrix_h.transform_matrix;
	h_mat_ws[2] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[2] * this->man_lp_matrix_h.transform_matrix;

	// 3. Transform back to the global reference frame
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{	v_loc << h_mat_ws[0](i,j), h_mat_ws[1](i,j), h_mat_ws[2](i,j);
			v_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * v_loc;
			
			h_mat_out[0](i,j) = v_glo(0);
			h_mat_out[1](i,j) = v_glo(1);
			h_mat_out[2](i,j) = v_glo(2);
		}
	}
}

void Manager::support_h_matrix_real_derivative2( const LonePair* lp, const double& sigma, const Eigen::Vector3d& Rij, /* workspace */ Eigen::Matrix4d (&h_mat_ws)[6], /* out */ Eigen::Matrix4d (&h_mat_out)[6] )
{
	this->man_lp_matrix_h.GetTransformationMatrix(Rij);     // get Transformation matrix ... saved : Eigen::Matrix4d this->man_lp_matrix_h.transform_matrix;
	Eigen::Matrix3d m_loc, m_glo;	// workspace
	for(int i=0;i<9;i++){ h_mat_ws[i].setZero(); }		// initialisation

	// XX
	h_mat_ws[0](0,0) = this->man_lp_matrix_h.real_ss_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](0,3) = this->man_lp_matrix_h.real_sz_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](3,0) = h_mat_ws[0](0,3);
	h_mat_ws[0](1,1) = this->man_lp_matrix_h.real_xx_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](2,2) = this->man_lp_matrix_h.real_yy_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[0](3,3) = this->man_lp_matrix_h.real_zz_grad2_xx_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	
	// XY
	h_mat_ws[1](1,2) = this->man_lp_matrix_h.real_xy_grad2_xy_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[1](2,1) = h_mat_ws[1](1,2);

	// XZ
	h_mat_ws[2](0,1) = this->man_lp_matrix_h.real_sx_grad2_xz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[2](1,0) = h_mat_ws[2](0,1);
	h_mat_ws[2](1,3) = this->man_lp_matrix_h.real_xz_grad2_xz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[2](3,1) = h_mat_ws[2](1,3);

	// ZZ
	h_mat_ws[5](0,0) = this->man_lp_matrix_h.real_ss_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[5](0,1) = this->man_lp_matrix_h.real_sz_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[5](1,0) = h_mat_ws[9](0,1);
	h_mat_ws[5](1,1) = this->man_lp_matrix_h.real_xx_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());
	h_mat_ws[5](2,2) = h_mat_ws[0](1,1);
	h_mat_ws[5](3,3) = this->man_lp_matrix_h.real_zz_grad2_zz_pc(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sigma,Rij.norm());

	// By orbital symmetries ...
	
	// YY ~ XX
	h_mat_ws[3] = h_mat_ws[0];	
	h_mat_ws[3](1,1) = h_mat_ws[0](2,2);	// YY xx = XX yy
	h_mat_ws[3](2,2) = h_mat_ws[0](1,1);	// YY yy = XX xx

	// YZ ~ XZ
	h_mat_ws[4](0,2) = h_mat_ws[4](2,0) = h_mat_ws[2](0,1);		// YZ sy = XZ sx
	h_mat_ws[4](2,3) = h_mat_ws[4](3,2) = h_mat_ws[2](1,3);		// YZ yz = XZ xz
	

	// TRANSFORMATION 1 : Using the local elements; compute equivalent elements (in the global) in the local reference frame
	for(int i=0;i<6;i++){ h_mat_ws[i] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[i] * this->man_lp_matrix_h.transform_matrix; }

	// TRANSFORMATION 2 : Transfrom back to the global reference frame (i,j) refer to basis function
	for(int i=0;i<4;i++)
	{	for(int j=0;j<4;j++)
		{
			m_loc << h_mat_ws[0](i,j), h_mat_ws[1](i,j), h_mat_ws[2](i,j),	// xx  xy  xz
				 h_mat_ws[1](i,j), h_mat_ws[3](i,j), h_mat_ws[4](i,j),	// yx  yy  yz
				 h_mat_ws[2](i,j), h_mat_ws[4](i,j), h_mat_ws[5](i,j);	// zx  yz  zz
	
			m_glo = this->man_lp_matrix_h.transform_matrix_shorthand.transpose() * m_loc * this->man_lp_matrix_h.transform_matrix_shorthand;

			h_mat_out[0](i,j) = m_glo(0,0); h_mat_out[1](i,j) = m_glo(0,1); h_mat_out[2](i,j) = m_glo(0,2);	// xx xy xz
							h_mat_out[3](i,j) = m_glo(1,1); h_mat_out[4](i,j) = m_glo(1,2);	//    yy yz
											h_mat_out[5](i,j) = m_glo(2,2);	//       zz

		}
	}

	return;
}

void Manager::set_h_matrix_real( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector, const bool is_first_scf )
{
	const std::string type_i = C.AtomList[i]->type;
	const std::string type_j = C.AtomList[j]->type;
	double lp_cf[4];
	double factor;		// Multiplication Factor ... controls species charges + etcs.
	double partial_e;	// Partial energy when 'i' is non-LP species and 'j' is LP density
	double real_pos[3];
	Eigen::Vector3d Rij;

	/* if 'lone' comes to the place 'i'    : compute h matrix 

	   else (i.e., comes to the place 'j') : compute the interaction energy -> based on the previous charge density shape (i.e., LP lone pair eigenvectors)
	*/

	/*
		LonePairD (in the central-sublattice) vs Classical Entities (core, shell, LPcore ... ) can be calculated only once ...? )

		Since,

		In the 'n'th (n!=1) SCF Cycle, the pre-calculated values are not changed, but only

		Classical Entities (in the central-sublattice) vs LonePairD (in the periodic images) are changing ... depending upon the changes in the LonePairD EigenVectors
	*/

	if( type_i == "lone" && type_j == "core" )	// LonePairD (central-sublattice, CSL) ----> Core(periodic image, PI)
	{
		#ifdef BOOST_MEMO	// INTEGRAL BOOST - SCF - Memoization scheme
		if( is_first_scf == true )
		#endif
		{
			LonePair* lp = static_cast<LonePair*>(C.AtomList[i]);
			// W.R.T Core
			Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;	// (Rj+T) - Ri ... Not using 'Ri - Rj - T' to get the right transformation // 'i' LPcore - 'j' core
			// Evaluation
			Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );	// output saved 'this->man_matrix4d_h_real_ws[1]'

			factor = 0.5 * lp->lp_charge * C.AtomList[j]->charge;			// Inverse Transformation .. mul 1/2 ... factor-out double counting
			this->LPC_H_Real[i][j][0] += factor * this->man_matrix4d_h_real_ws[1];
		}
	}

	if( type_i == "core" && type_j == "lone" )	// Core (CSL) ----> LonePairD (PI)
	{
		// Get 'j' lone pair cation & its eigenvectors of the ground-state
		LonePair* lp = static_cast<LonePair*>(C.AtomList[j]);
		lp_cf[0] = lp->lp_eigensolver.eigenvectors()(0,lp->lp_gs_index).real();	// s
		lp_cf[1] = lp->lp_eigensolver.eigenvectors()(1,lp->lp_gs_index).real();	// px
		lp_cf[2] = lp->lp_eigensolver.eigenvectors()(2,lp->lp_gs_index).real();	// py
		lp_cf[3] = lp->lp_eigensolver.eigenvectors()(3,lp->lp_gs_index).real();	// pz

		// W.R.T LP Density
		Rij = C.AtomList[i]->cart - ( C.AtomList[j]->cart + TransVector );	// Ri - ( Rj + T ) where 'i' core & 'j' LPcore (in a periodic image)
		// Evalulation
		Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

		factor = 0.5 * C.AtomList[i]->charge * lp->lp_charge;
		this->man_matrix4d_h_real_ws[1] = factor * this->man_matrix4d_h_real_ws[1];		// POSSIBLE MEMOIZATION ... (may be difficult ... since 'j' LP depends on lattice translation 'T')

		// Calculate Energy Contribution by the given LP density in the periodic image and a point charge in the central sublattice
		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_real_ws[1](ii,jj); }}			// PartialE ... Process Required
		// Save LonePair Density Energy ...
		C.lp_real_energy += partial_e;
	}

	if( type_i == "lone" && type_j == "shel" )	// LonePairD (CSL) ----> Shell (PI)
	{
		#ifdef BOOST_MEMO
		if( is_first_scf == true )
		#endif
		{
			LonePair* lp = static_cast<LonePair*>(C.AtomList[i]);

			// <1> W.R.T CorePart (PI)
			Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;	// (Rj+T) - Ri ... Ri(LPcore) ---> Rj+T(shell CorePart);
			// Evalulation
			Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

			factor = 0.5 * lp->lp_charge * C.AtomList[j]->charge;
			this->LPC_H_Real[i][j][0] += factor * this->man_matrix4d_h_real_ws[1];		// [0] for Core
			
			// <2> W.R.T ShelPart (PI)
			Rij = ( static_cast<Shell*>(C.AtomList[j])->shel_cart + TransVector ) - C.AtomList[i]->cart;	// (Rj+T) - Ri ... Ri(LPcore) ---> Rj+T(ShellPart);
			// Evalulation
			Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );
			
			factor = 0.5 * lp->lp_charge * static_cast<Shell*>(C.AtomList[j])->shel_charge;
			this->LPC_H_Real[i][j][1] += factor * this->man_matrix4d_h_real_ws[1];		// [1] for Shell
		}
	}

	if( type_i == "shel" && type_j == "lone" )	// Shell (CSL) ----> LonePairD (PI)
	{
		// Get 'j' lone pair cation & its eigenvectors of the ground-state
		LonePair* lp = static_cast<LonePair*>(C.AtomList[j]);
		lp_cf[0] = lp->lp_eigensolver.eigenvectors()(0,lp->lp_gs_index).real();	// s
		lp_cf[1] = lp->lp_eigensolver.eigenvectors()(1,lp->lp_gs_index).real();	// px
		lp_cf[2] = lp->lp_eigensolver.eigenvectors()(2,lp->lp_gs_index).real();	// py
		lp_cf[3] = lp->lp_eigensolver.eigenvectors()(3,lp->lp_gs_index).real();	// pz

		// <1> Core W.R.T LP Density 
		Rij = C.AtomList[i]->cart - ( C.AtomList[j]->cart + TransVector );	// Ri - ( Rj + T ) where 'i' core & 'j' LPcore (in a periodic image)
		// Evalulation
		Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

		factor = 0.5 * C.AtomList[i]->charge * lp->lp_charge;
		this->man_matrix4d_h_real_ws[1] = factor * this->man_matrix4d_h_real_ws[1];

		// Calculation Energy Contribution by the given LP density in the periodic image and the core in the central sublattice
		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_real_ws[1](ii,jj); }}			// PartialE ... Process Required
		// Save LonePair Density Energy ...
		C.lp_real_energy += partial_e;

		// <2> Shel W.R.T LP Density
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - ( C.AtomList[j]->cart + TransVector );
		// Evalulation
		Manager::support_h_matrix_real( lp, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

		factor = 0.5 * static_cast<Shell*>(C.AtomList[i])->shel_charge * lp->lp_charge;
		this->man_matrix4d_h_real_ws[1] = factor * this->man_matrix4d_h_real_ws[1];

		// Calculation Energy Contribution by the given LP density in the periodic image and the shel in the central sublattice
		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_real_ws[1](ii,jj); }}			// PartialE ... Process Required
		// Save LonePair Density Energy ...
		C.lp_real_energy += partial_e;
	}

	if( type_i == "lone" && type_j == "lone" )	// i == j will not get caught when h=k=l=0 by 'if' of its wrapper
	{
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
		LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);
	
		#ifdef BOOST_MEMO
		if( is_first_scf == true )
		#endif
		{
			// <1> LP(i) Density (CSL) vs LP(j) Core (PI)	.... analogy Shell - Core
			Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;	// LP(i)('in the central sublattice')  -> LP(j)('in the periodic image') core
			// Evaluation
			Manager::support_h_matrix_real( lpi, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

			factor = 0.5 * lpi->lp_charge * C.AtomList[j]->charge;			// LP(i) charge * LP(j) core charge
			this->LPC_H_Real[i][j][0] += factor * this->man_matrix4d_h_real_ws[1];		// LPcore treated as a classical entity ... saved in 'LPC_H_...'
		}

		// <2> LP(i) Core (CSL) vs LP(j) Density (PI)	.... analogy Core - Shell
		Rij = C.AtomList[i]->cart - ( C.AtomList[j]->cart + TransVector );	// LP(j)('in the periodic image') -> LP(j)('in the central sublattice')
		// Get LP(j) Density eigenvectors 
		lp_cf[0] = lpj->lp_eigensolver.eigenvectors()(0,lpj->lp_gs_index).real();	// s
		lp_cf[1] = lpj->lp_eigensolver.eigenvectors()(1,lpj->lp_gs_index).real();	// px
		lp_cf[2] = lpj->lp_eigensolver.eigenvectors()(2,lpj->lp_gs_index).real();	// py
		lp_cf[3] = lpj->lp_eigensolver.eigenvectors()(3,lpj->lp_gs_index).real();	// pz
		// Evalulation
		Manager::support_h_matrix_real( lpj, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

		factor = 0.5 * C.AtomList[i]->charge * lpj->lp_charge;			// LP(i) core charge * LP(j) charge

		// Calculation Energy Contribution by the given LP density in the periodic image and the LP core in the central sublattice
		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++) { partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_real_ws[1](ii,jj); }}			// PartialE ... Process Required
		C.lp_real_energy += partial_e;

		// <3> LP(i) Density vs LP(j) Density	....	ananlogy Shell - Shell

		////	////	////	////	////	////	////	////	////	////	////	////	////	////
		////	3.A Monopolar Term	

		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;	// LP(i) (CSL) ----> LP(j) (PI)		
		// Evalulation
		Manager::support_h_matrix_real( lpi, C.sigma, Rij, this->man_matrix4d_h_real_ws[0], this->man_matrix4d_h_real_ws[1] );

		factor = 0.5 * lpi->lp_charge * lpj->lp_charge;
		this->LPLP_H_Real[i][j] += factor * this->man_matrix4d_h_real_ws[1];
		////	****************************************************************************************************

		////	////	////	////	////	////	////	////	////	////	////	////	////	////
		////	3.B Dipolar Term ... depends on the LP density of 'j'th LP

		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;	// LP(i) (CSL) ----> LP(j) (PI)
		// Evalulation
		Manager::support_h_matrix_real_derivative( lpi, C.sigma, Rij, this->man_matrix4d_h_real_derivative_ws, this->man_matrix4d_h_real_derivative_out );
		// decltype(*_out) - type Eigen::Matrix4d[3] where [0-2] - [x],[y],[z]
	 
		// Get LP(j) Density eigenvectors - Re-use 'lp_cf[0-3]'
		// factor - Re-use 'factor = 0.5 * lpi->lp_charge * lpj->lp_charge;' above
		real_pos[0] = 2.*lp_cf[0]*lp_cf[1]*lpj->lp_real_position_integral;	// 2 cs cpx	// N.B. 'lp_cf' eigenvectors of 'j' LP Density
		real_pos[1] = 2.*lp_cf[0]*lp_cf[2]*lpj->lp_real_position_integral;	// 2 cs cpy
		real_pos[2] = 2.*lp_cf[0]*lp_cf[3]*lpj->lp_real_position_integral;	// 2 cs cpz

		this->LPLP_H_Real[i][j] += factor * ( real_pos[0] * this->man_matrix4d_h_real_derivative_out[0] + real_pos[1] * this->man_matrix4d_h_real_derivative_out[1] + real_pos[2] * this->man_matrix4d_h_real_derivative_out[2] );
		//MAKE SURE THE SIGN IS CORRECT!!!

		// * Save Dipolar Terms ... For Geometric Derivative Calculation Later...
		this->LPLP_H_Real_Aux[i][j][0] += factor * lpj->lp_real_position_integral * this->man_matrix4d_h_real_derivative_out[0];	// factor is just charges * 0.5 ... 1/2 * qi * qj
		this->LPLP_H_Real_Aux[i][j][1] += factor * lpj->lp_real_position_integral * this->man_matrix4d_h_real_derivative_out[1];
		this->LPLP_H_Real_Aux[i][j][2] += factor * lpj->lp_real_position_integral * this->man_matrix4d_h_real_derivative_out[2];	// must keep the form before 'b' molecular orbtial coefficients applied

		/* 
			SUM(l,s) Cbl*Cbs*<ua| dxHab <lb|rbx|sb> + dyHab <lb|rby|sb> + dzHab <lb|rbz|sb> |va> 
			=  <ua|dxHab|va>*f*p * (2*Cbs*Cbx) + <ua|dyHab|va>*f*p * (2*Cbs*Cby) + <ua|dzHab|va>*f*p * (2*Cbs*Cbz)

			Save 1 - <ua|dxHab|va> * f * p
			Save 2 - <ua|dyHab|va> * f * p
			Save 3 - <ua|dzHab|va> * f * p

			The savings will be used for calculating evec derivatives.. w.r.t. x (atom coordinates), G (reciprocal lattice vectors), V (Cell volume)
		*/

#ifdef LPLP_CHECK
std::cout << "LPLP+CHECK" << std::endl;
std::cout << "Vector Rij" << std::endl;
ShowVector(Rij);
std::cout << "Ri" << std::endl;
ShowVector(C.AtomList[i]->cart);
std::cout << "Rj" << std::endl;
ShowVector(C.AtomList[j]->cart);
std::cout << "T vector" << std::endl;
ShowVector(TransVector);

std::cout << "Monopole Term" << std::endl;
ShowMatrix(this->man_matrix4d_h_real_ws[1]);
std::cout << "Dipolar X\n";
ShowMatrix(this->man_matrix4d_h_real_derivative_out[0]);
std::cout << "Dipolar Y\n";
ShowMatrix(this->man_matrix4d_h_real_derivative_out[1]);
std::cout << "Dipolar Z\n";
ShowMatrix(this->man_matrix4d_h_real_derivative_out[2]);
std::cout << "------------------------------------------------- TEST Rij\n";

exit(1);
#endif

	}	// END lp-lp interaction if-statement
	return;
}

void Manager::CoulombLonePairReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector, const bool is_first_scf )
{
	double Qi,Qj;
	Eigen::Vector3d Rij;
        // TransVector = h*a + k*b + l*c
        // Rij         = Ai.r - Aj.r - TransVector;

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		if( is_first_scf == true )
		{
			// LonePair Core - Core
			Qi  = C.AtomList[i]->charge;	// Get Lone CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		}

	}
	
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// Core - LonePair Core
			Qi  = C.AtomList[i]->charge;	// Get CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get Lone CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" )
	{
		if( is_first_scf == true )
		{
			// LonePair Core - Shell Core
			Qi  = C.AtomList[i]->charge;	// Get Lone CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get Shel CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;

			// LonePair Core - Shell Shell
			Qi  = C.AtomList[i]->charge;
			Qj  = static_cast<Shell*>(C.AtomList[j])->shel_charge;
			Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// Shell Core - LonePair Core
			Qi  = C.AtomList[i]->charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;

			// Shell Shell - LonePair Core
			Qi  = static_cast<Shell*>(C.AtomList[i])->shel_charge;
			Qj  = C.AtomList[j]->charge;
			Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// LonePair Core - LonePair Core
			Qi  = C.AtomList[i]->charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
			C.mono_real_energy += 0.5*(Qi*Qj)/Rij.norm() * erfc(Rij.norm()/C.sigma) * C.TO_EV;
		}
	}
	// Accumulating LonePair <---> LonePair / Core / Shell Interactions; i.e., setting up the LonePair Hamiltonian Matrices
	Manager::set_h_matrix_real( C, i, j, TransVector, is_first_scf );
}

void Manager::CoulombLonePairSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector, const bool is_first_scf )	// self energy in the reciprocal space
{
	double Qi,Qj;
	double factor;
	double intact;
        // TransVector = h*a + k*b + l*c

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )	// SelfEnergy by LonePair Cores
	{
		if( is_first_scf == true )		// LPcore - LPcore Self Interaction
		{	// core self
			Qi  = C.AtomList[i]->charge;
			Qj  = C.AtomList[j]->charge;
			C.mono_reci_self_energy += -0.5*(Qi*Qj)*2./C.sigma/sqrt(M_PI) * C.TO_EV;
		}

		// (1) - LP Electron Self Energy (does not require to solve any integrals) - ... Category: LPLP Interaction
		Qi = static_cast<LonePair*>(C.AtomList[i])->lp_charge;
		Qj = static_cast<LonePair*>(C.AtomList[j])->lp_charge;
		intact = -0.5*(Qi*Qj)*2./C.sigma/sqrt(M_PI) * C.TO_EV;
		this->man_matrix4d_h_reci_self_ws.setZero();
		this->man_matrix4d_h_reci_self_ws(0,0) = this->man_matrix4d_h_reci_self_ws(1,1) = this->man_matrix4d_h_reci_self_ws(2,2) = this->man_matrix4d_h_reci_self_ws(3,3) = intact;
		this->LPLP_H_Reci[i][j] += this->man_matrix4d_h_reci_self_ws;

		#ifdef BOOST_MEMO
		if( is_first_scf == true )
		#endif
		{
			// (2) - LP Electron::Core, Core::LP Electron Self Energy -- Must be counted twice
			LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
			Qi = lpi->lp_charge;
			Qj = C.AtomList[j]->charge;
			//factor = -0.5 * (Qi*Qj) * 2.;
			factor = -(Qi*Qj);
			this->man_matrix4d_h_reci_self_ws.setZero();
			this->man_matrix4d_h_reci_self_ws(0,0) = this->man_lp_matrix_h.reci_self_integral_ss(lpi->lp_r,lpi->lp_r_s_function,lpi->lp_r_p_function,C.sigma);	// ss		  Component
			this->man_matrix4d_h_reci_self_ws(1,1) = this->man_lp_matrix_h.reci_self_integral_xx(lpi->lp_r,lpi->lp_r_s_function,lpi->lp_r_p_function,C.sigma);	// xx == yy == zz Component
			this->man_matrix4d_h_reci_self_ws(2,2) = this->man_matrix4d_h_reci_self_ws(3,3) = this->man_matrix4d_h_reci_self_ws(1,1);
			//this->LPLP_H_Reci[i][j] += factor * this->man_matrix4d_h_reci_self_ws;
			this->LPC_H_Reci[i][j][0] += factor * this->man_matrix4d_h_reci_self_ws;

			this->LPC_H_Reci_Self[i] = factor * this->man_matrix4d_h_reci_self_ws;
		}
		else	//DEBUGGING_FEB_20_2023
		{
			LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
			Qi = lpi->lp_charge;
			Qj = C.AtomList[j]->charge;
			factor = -0.5 * (Qi*Qj) * 2.;
			double en = 0.;
			double lpi_cf[4];
			lpi->GetEvecGS(lpi_cf);
			for(int u=0;u<4;u++)
			{	for(int v=0;v<4;v++)
				{
					en += lpi_cf[u]*lpi_cf[v]*factor*this->LPC_H_Reci_Self[i](u,v);
				}
			}
			std::cout << std::endl;
			std::cout << "LPC Self (" << i << ") ---------------------------------------------\n";
			printf("%20.12lf\n",en);
			std::cout << "--------------------------------------------------------------------\n";
		}


	}

	return;
}

void Manager::support_h_matrix_reci( const LonePair* lp, const Eigen::Vector3d& G, /* workspace */ Eigen::Matrix4d (&h_mat_ws)[2], /* out */ Eigen::Matrix4d (&h_mat_out)[2] )
{
	this->man_lp_matrix_h.GetTransformationMatrix(G);	// unit G (Angs^-1)
	h_mat_ws[0].setZero();  h_mat_ws[1].setZero();
	h_mat_out[0].setZero(); h_mat_out[1].setZero();

	const double Gnorm = G.norm();
	// Evaluation - Reciprocal Cosine Part
	h_mat_ws[0](0,0) = this->man_lp_matrix_h.reci_ss_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,Gnorm);
	h_mat_ws[0](1,1) = this->man_lp_matrix_h.reci_xx_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,Gnorm);
	h_mat_ws[0](2,2) = h_mat_ws[0](1,1);
	h_mat_ws[0](3,3) = this->man_lp_matrix_h.reci_zz_cos(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,Gnorm);
	// Evaluation - Reciprocal Sine   Part
	h_mat_ws[1](0,3) = this->man_lp_matrix_h.reci_sz_sin(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,Gnorm);
	h_mat_ws[1](3,0) = h_mat_ws[1](0,3);

	// Inverse Transformation
	h_mat_out[0] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[0] * this->man_lp_matrix_h.transform_matrix;	// Cos
	h_mat_out[1] = this->man_lp_matrix_h.transform_matrix.transpose() * h_mat_ws[1] * this->man_lp_matrix_h.transform_matrix;	// Sin
}

void Manager::set_h_matrix_reci( Cell& C, const int i, const int j, const Eigen::Vector3d& G, const bool is_first_scf )
{
	const std::string type_i = C.AtomList[i]->type;
	const std::string type_j = C.AtomList[j]->type;
	double lp_cf[4];		// Eigenvector storage
	double partial_e;		// Partial energy when 'i' is non-LP species and 'j' is LP density

	double factor, intact[4];	// factor : halved leading term / intact : Cos(G.Rij), Sin(G.Rij), rests->for LPLP
	double Qi,Qj;
	const double g_sqr = G.adjoint()*G;	//													// ERROR!!!
	Eigen::Vector3d Rij;	// Saving Real Space Rij
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;

	if( type_i == "lone" && type_j == "core" )	// LonePairD (central-sublattice, CSL) ----> Core(periodic image, PI)
	{
		#ifdef BOOST_MEMO	
		if( is_first_scf == true )
		#endif
		{	// 'i' LP in CSL
			LonePair* lp = static_cast<LonePair*>(C.AtomList[i]);

			Qi  = lp->lp_charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

			factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
			intact[0] = cos(Rij.adjoint()*G); // (Ri-Rj).G
			intact[1] = sin(Rij.adjoint()*G); // (Ri-Rj).G
			
			// Evaluation
			Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine	// return unit ... dimensionless
			this->LPC_H_Reci[i][j][0] += factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);
		}
	}

	if( type_i == "core" && type_j == "lone" )	// Core (CSL) ----> LonePairD (PI)
	{
		// 'i' core in CSL and 'j' LP in PI
		LonePair* lp = static_cast<LonePair*>(C.AtomList[j]);

		lp_cf[0] = lp->lp_eigensolver.eigenvectors()(0,lp->lp_gs_index).real();	// s
		lp_cf[1] = lp->lp_eigensolver.eigenvectors()(1,lp->lp_gs_index).real();	// px
		lp_cf[2] = lp->lp_eigensolver.eigenvectors()(2,lp->lp_gs_index).real();	// py
		lp_cf[3] = lp->lp_eigensolver.eigenvectors()(3,lp->lp_gs_index).real();	// pz

		Qi  = C.AtomList[i]->charge;
		Qj  = lp->lp_charge;
		Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;

		factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term	// Unit (eV)
		intact[0] = cos(Rij.adjoint()*G); // (Rj-Ri).G
		intact[1] = sin(Rij.adjoint()*G);

		// Evaluation
		Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
		this->man_matrix4d_h_reci_ws[0] = factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);

		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_reci_ws[0](ii,jj); }}			// PartialE ... Process Required
		C.lp_reci_energy += partial_e;
	}

	if( type_i == "lone" && type_j == "shel" )	// LonePairD (CSL) ----> Shell (PI)
	{	
		#ifdef BOOST_MEMO
		if( is_first_scf == true )
		#endif
		{
			LonePair* lp = static_cast<LonePair*>(C.AtomList[i]);
			// <1> W.R.T CorePart (PI)
			Qi  = lp->lp_charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

			factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
			intact[0] = cos(Rij.adjoint()*G);
			intact[1] = sin(Rij.adjoint()*G);

			// Evaluation
			Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
			this->LPC_H_Reci[i][j][0] += factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);

			// <2> W.R.T ShelPart (PI)
			Qi  = lp->lp_charge;
			Qj  = static_cast<Shell*>(C.AtomList[j])->shel_charge;
			Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

			factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
			intact[0] = cos(Rij.adjoint()*G);
			intact[1] = sin(Rij.adjoint()*G);

			// Evaluation
			//Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
			this->LPC_H_Reci[i][j][1] += factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);
		}
	}

	if( type_i == "shel" && type_j == "lone" )	// Shell (CSL) ----> LonePairD (PI)
	{
		// 'i' shel in CSL and 'j' LP in PI
		LonePair* lp = static_cast<LonePair*>(C.AtomList[j]);

		lp_cf[0] = lp->lp_eigensolver.eigenvectors()(0,lp->lp_gs_index).real();	// s
		lp_cf[1] = lp->lp_eigensolver.eigenvectors()(1,lp->lp_gs_index).real();	// px
		lp_cf[2] = lp->lp_eigensolver.eigenvectors()(2,lp->lp_gs_index).real();	// py
		lp_cf[3] = lp->lp_eigensolver.eigenvectors()(3,lp->lp_gs_index).real();	// pz

		// <1> Core W.R.T LP Density 
		Qi  = C.AtomList[i]->charge;
		Qj  = lp->lp_charge;
		Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;

		factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
		intact[0] = cos(Rij.adjoint()*G);
		intact[1] = sin(Rij.adjoint()*G);

		// Evaluation
		Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
		this->man_matrix4d_h_reci_ws[0] = factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);

		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_reci_ws[0](ii,jj); }}			// PartialE ... Process Required
		C.lp_reci_energy += partial_e;

		// <2> Shel W.R.T LP Density
		Qi  = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj  = lp->lp_charge;
		Rij = C.AtomList[j]->cart - static_cast<Shell*>(C.AtomList[i])->shel_cart;

		factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
		intact[0] = cos(Rij.adjoint()*G);
		intact[1] = sin(Rij.adjoint()*G);

		// Evaluation
		//Manager::support_h_matrix_reci( lp, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
		this->man_matrix4d_h_reci_ws[0] = factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);

		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_reci_ws[0](ii,jj); }}			// PartialE ... Process Required
		C.lp_reci_energy += partial_e;
	}

	if( type_i == "lone" && type_j == "lone" )	// i == j will not get caught when h=k=l=0 by 'if' of its wrapper
	{
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
		LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);

		lp_cf[0] = lpj->lp_eigensolver.eigenvectors()(0,lpj->lp_gs_index).real();	// s
		lp_cf[1] = lpj->lp_eigensolver.eigenvectors()(1,lpj->lp_gs_index).real();	// px
		lp_cf[2] = lpj->lp_eigensolver.eigenvectors()(2,lpj->lp_gs_index).real();	// py
		lp_cf[3] = lpj->lp_eigensolver.eigenvectors()(3,lpj->lp_gs_index).real();	// pz

		#ifdef BOOST_MEMO
		if( is_first_scf == true )
		#endif
		{
			// LP   Core	
			Qi  = lpi->lp_charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

			factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
			intact[0] = cos(Rij.adjoint()*G);
			intact[1] = sin(Rij.adjoint()*G);
			// Evaluation
			Manager::support_h_matrix_reci( lpj, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
			this->LPC_H_Reci[i][j][0] += factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);
		}

		// (2) Core LP
		Qi  = C.AtomList[i]->charge;
		Qj  = lpj->lp_charge;
		Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;

		factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
		intact[0] = cos(Rij.adjoint()*G);
		intact[1] = sin(Rij.adjoint()*G);

		// Evaluation
		#ifdef BOOST_MEMO
		Manager::support_h_matrix_reci( lpj, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine - it doesn't really matter using const LonePair* (1st arg) place 'lpi' or 'lpj'
		#endif

		this->man_matrix4d_h_reci_ws[0] = factor*(intact[0]*this->man_matrix4d_h_reci_out[0] - intact[1]*this->man_matrix4d_h_reci_out[1]);

		partial_e = 0.;
		for(int ii=0;ii<4;ii++){ for(int jj=0;jj<4;jj++){ partial_e += lp_cf[ii] * lp_cf[jj] * this->man_matrix4d_h_reci_ws[0](ii,jj); }}			// PartialE ... Process Required
		// Save LonePair Density Energy ...
		C.lp_reci_energy += partial_e;

		// (3) LP   LP
		Qi  = lpi->lp_charge;
		Qj  = lpj->lp_charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		factor    = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;	// halved leading term
		intact[0] = cos(Rij.adjoint()*G);
		intact[1] = sin(Rij.adjoint()*G);

/*
		// Evaluation
		//Manager::support_h_matrix_reci( lpj, G, this->man_matrix4d_h_reci_ws, this->man_matrix4d_h_reci_out );	// man_matrix4d_h_reci_ws/out[0-1] : 0 cosine 1 sine
		intact[2] = intact[3] = 0.;
		for(int ii=0;ii<4;ii++)
		{	for(int jj=0;jj<4;jj++)
			{	intact[2] += lp_cf[ii]*lp_cf[jj]*this->man_matrix4d_h_reci_out[0](ii,jj);	// Cj
				intact[3] += lp_cf[ii]*lp_cf[jj]*this->man_matrix4d_h_reci_out[1](ii,jj);	// Sj
			}
		}
		
		for(int ii=0;ii<4;ii++)
		{	for(int jj=0;jj<4;jj++)
			{
				this->LPLP_H_Reci[i][j](ii,jj) += factor * ( intact[0] * (this->man_matrix4d_h_reci_out[0](ii,jj)*intact[2] + this->man_matrix4d_h_reci_out[1](ii,jj)*intact[3])
									   + intact[1] * (this->man_matrix4d_h_reci_out[0](ii,jj)*intact[3] - this->man_matrix4d_h_reci_out[1](ii,jj)*intact[2]) );		
			}
		}
*/
		// Equivalent expression with above
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	for(int l=0;l<4;l++)
				{	for(int s=0;s<4;s++)
					{ this->LPLP_H_Reci[i][j](u,v) += factor *(lp_cf[s]*lp_cf[l]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
											             +this->man_matrix4d_h_reci_out[1](l,s)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v))));
					}// s
				}// l
			}// v
		}// u


		/*
 			H_LPLP_Reci[a][b](u,v) += f * cb_l * cb_s * ( CosInt_b(l,s) * ( cos(G.Rab) * CosInt_a(u,v) - sin(G.Rab) * SinInt_a(u,v) )
								     +SinInt_b(l,s) * ( cis(G.Rab) * SinInt_a(u,v) + sin(G.Rab) * CosInt_a(u,v) ) );

			indices .. l and s not looped through, i.e., they are fixed
			
			Save - 1  : (l,s) = (s,s)
			Save - 2  : (l,s) = (s,x)
			Save - 3  : (l,s) = (s,y)
			Save - 4  : (l,s) = (s,z)
			Save - 5  : (l,s) = (x,x)
			Save - 6  : (l,s) = (x,y)
			Save - 7  : (l,s) = (x,z)
			Save - 8  : (l,s) = (y,y)
			Save - 9  : (l,s) = (y,z)
			Save - 10 : (l,s) = (z,z)

			The savings will be used for calculating evec derivatives.. w.r.t. x (atom coordinates), G (reciprocal lattice vectors), V (Cell volume)
		*/
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	// SS
				this->LPLP_H_Reci_Aux[i][j][0](u,v) += factor * (this->man_matrix4d_h_reci_out[0](0,0)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](0,0)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// SX * 2
				this->LPLP_H_Reci_Aux[i][j][1](u,v) += factor * (this->man_matrix4d_h_reci_out[0](0,1)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](0,1)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// SY * 2
				this->LPLP_H_Reci_Aux[i][j][2](u,v) += factor * (this->man_matrix4d_h_reci_out[0](0,2)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](0,2)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// SZ * 2
				this->LPLP_H_Reci_Aux[i][j][3](u,v) += factor * (this->man_matrix4d_h_reci_out[0](0,3)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](0,3)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// XX
				this->LPLP_H_Reci_Aux[i][j][4](u,v) += factor * (this->man_matrix4d_h_reci_out[0](1,1)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](1,1)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// XY * 2
				this->LPLP_H_Reci_Aux[i][j][5](u,v) += factor * (this->man_matrix4d_h_reci_out[0](1,2)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](1,2)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// XZ * 2
				this->LPLP_H_Reci_Aux[i][j][6](u,v) += factor * (this->man_matrix4d_h_reci_out[0](1,3)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](1,3)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// YY
				this->LPLP_H_Reci_Aux[i][j][7](u,v) += factor * (this->man_matrix4d_h_reci_out[0](2,2)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](2,2)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// YZ * 2
				this->LPLP_H_Reci_Aux[i][j][8](u,v) += factor * (this->man_matrix4d_h_reci_out[0](2,3)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](2,3)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));
				// ZZ * 2
				this->LPLP_H_Reci_Aux[i][j][9](u,v) += factor * (this->man_matrix4d_h_reci_out[0](3,3)*(intact[0]*this->man_matrix4d_h_reci_out[0](u,v)-intact[1]*this->man_matrix4d_h_reci_out[1](u,v))
										+this->man_matrix4d_h_reci_out[1](3,3)*(intact[0]*this->man_matrix4d_h_reci_out[1](u,v)+intact[1]*this->man_matrix4d_h_reci_out[0](u,v)));

			}
		}
	}
	return;
}

void Manager::CoulombLonePairReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector, const bool is_first_scf )
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		if( is_first_scf == true )
		{
			// LonePair Core - Core
			Qi  = C.AtomList[i]->charge;	// Get Lone CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		}
	}
	
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// Core - LonePair Core
			Qi  = C.AtomList[i]->charge;	// Get CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get Lone CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" )
	{
		if( is_first_scf == true )
		{
			// LonePair Core - Shell Core
			Qi  = C.AtomList[i]->charge;	// Get Lone CoreCharge
			Qj  = C.AtomList[j]->charge;	// Get Shel CoreCharge
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;

			// LonePair Core - Shell Shell
			Qi  = C.AtomList[i]->charge;
			Qj  = static_cast<Shell*>(C.AtomList[j])->shel_charge;
			Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// Shell Core - LonePair Core
			Qi  = C.AtomList[i]->charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;

			// Shell Shell - LonePair Core
			Qi  = static_cast<Shell*>(C.AtomList[i])->shel_charge;
			Qj  = C.AtomList[j]->charge;
			Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		}

	}

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )
	{
		if( is_first_scf == true )
		{
			// LonePair Core - LonePair Core
			Qi  = C.AtomList[i]->charge;
			Qj  = C.AtomList[j]->charge;
			Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
			C.mono_reci_energy += (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * cos( TransVector.adjoint()*Rij ) * C.TO_EV;
		}
	}

	Manager::set_h_matrix_reci( C, i, j, TransVector, is_first_scf );
}































////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

////	LonePair_Derivative

////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

void Manager::InitialiseLonePairCalculation_Derivatives( Cell& C )
{
	for(int i=0;i<C.NumberOfAtoms;i++)
	{
		for(int j=0;j<C.NumberOfAtoms;j++)
		{	// Real LP-MM,LPcore
			LPC_H_Real_Derivative[i][j][0][0].setZero(); LPC_H_Real_Derivative[i][j][0][1].setZero(); LPC_H_Real_Derivative[i][j][0][2].setZero();
			LPC_H_Real_Derivative[i][j][1][0].setZero(); LPC_H_Real_Derivative[i][j][1][1].setZero(); LPC_H_Real_Derivative[i][j][1][2].setZero();
			// Real LP-LP
			LPLP_H_Real_Derivative[i][j][0].setZero(); LPLP_H_Real_Derivative[i][j][1].setZero(); LPLP_H_Real_Derivative[i][j][2].setZero();
			// Reci LP-MM,LPcore
			LPC_H_Reci_Derivative[i][j][0][0].setZero(); LPC_H_Reci_Derivative[i][j][0][1].setZero(); LPC_H_Reci_Derivative[i][j][0][2].setZero();
			LPC_H_Reci_Derivative[i][j][1][0].setZero(); LPC_H_Reci_Derivative[i][j][1][1].setZero(); LPC_H_Reci_Derivative[i][j][2][2].setZero();
			// Reci LP-LP
			LPLP_H_Reci_Derivative[i][j][0].setZero(); LPLP_H_Reci_Derivative[i][j][1].setZero(); LPLP_H_Reci_Derivative[i][j][2].setZero();
		}
		// Self
		LP_H_Self_Derivative[i][0].setZero(); LP_H_Self_Derivative[i][1].setZero(); LP_H_Self_Derivative[i][2].setZero();
	}
	return;
}

/*
 * Sign Guide Line : any derivatives are calculated w.r.t. (j) of (i) ----> (j) ... i.e., (i) is assumed to be placed at the local centre or the local origin
 */
void Manager::set_h_matrix_real_derivative( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	const std::string type_i = C.AtomList[i]->type;
	const std::string type_j = C.AtomList[j]->type;
	double lpi_cf[4], lpj_cf[4];
	double factor;
	
	Eigen::Vector3d Rij;
	Eigen::Vector3d D; D.setZero();
	
	if( type_i == "lone" && type_j == "core" )
	{	
		LonePair* lp = static_cast<LonePair*>(C.AtomList[i]);
		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart; // ( Rj + T ) - Ri ... (i) - at the relative origin ----------> (j) 
		// H Derivative Evalulation ... displacements on (j)
		Manager::support_h_matrix_real_derivative(lp,C.sigma,Rij,this->man_matrix4d_h_real_derivative_ws,this->man_matrix4d_h_real_derivative_out);	// this derivative is w.r.t -----> target i.e., w.r.t 'core' (j)
		factor = 0.5 * lp->lp_charge * C.AtomList[j]->charge;

		// reason for '-' sign: the derivative is w.r.t the displs of 'j'. To make it for 'i' ... sign must be inverted
		LPC_H_Real_Derivative[i][j][0][0] -= factor * this->man_matrix4d_h_real_derivative_out[0];	
		LPC_H_Real_Derivative[i][j][0][1] -= factor * this->man_matrix4d_h_real_derivative_out[1];	
		LPC_H_Real_Derivative[i][j][0][2] -= factor * this->man_matrix4d_h_real_derivative_out[2];	// pair geometric derivatives must be considered later
		// eigenvectors of (i) use									
		
		// derivative update
		this->DerivativeH[0] = -factor * this->man_matrix4d_h_real_derivative_out[0];
		this->DerivativeH[1] = -factor * this->man_matrix4d_h_real_derivative_out[1];
		this->DerivativeH[2] = -factor * this->man_matrix4d_h_real_derivative_out[2];
		lp->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;
		

	}
	if( type_i == "core" && type_j == "lone" )
	{
		LonePair* lp = static_cast<LonePair*>(C.AtomList[j]);
		Rij = C.AtomList[i]->cart - ( C.AtomList[j]->cart + TransVector ); // Ri - ( Rj + T ) ... (j) - at the relative origin --------> (i)
		// Derivative Evaluation ... displacements on (i)
		Manager::support_h_matrix_real_derivative(lp,C.sigma,Rij,this->man_matrix4d_h_real_derivative_ws,this->man_matrix4d_h_real_derivative_out);	// this derivative is w.r.t -----> target i.e., w.r.t 'core' (i)
		factor = 0.5 * lp->lp_charge * C.AtomList[i]->charge;

		// 'H' derivative w.r.t displs (j) therefore sign '-'
		LPC_H_Real_Derivative[j][i][0][0] -= factor * this->man_matrix4d_h_real_derivative_out[0];
		LPC_H_Real_Derivative[j][i][0][1] -= factor * this->man_matrix4d_h_real_derivative_out[1];
		LPC_H_Real_Derivative[j][i][0][2] -= factor * this->man_matrix4d_h_real_derivative_out[2];
		// eigenvectors of (j) use
		
		// derivative update
		this->DerivativeH[0] = -factor * this->man_matrix4d_h_real_derivative_out[0];
		this->DerivativeH[1] = -factor * this->man_matrix4d_h_real_derivative_out[1];
		this->DerivativeH[2] = -factor * this->man_matrix4d_h_real_derivative_out[2];
		lp->GetEvecGS(lpj_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[j]->cart_gd += D;
		C.AtomList[i]->cart_gd -= D;
	}

	if( type_i == "lone" && type_j == "shel " )
	{	// NotImplemented
	}
	if( type_i == "shel" && type_j == "lone " )
	{	// NotImplemented
	}

	if( type_i == "lone" && type_j == "lone" )
	{
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
		LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);
		lpi->GetEvecGS( lpi_cf ); lpj->GetEvecGS( lpj_cf );

		/* 1-1 LP(i) LP -----> LP(j) CORE */
		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;
		// Derivative Evaluation ... displacement on (j)
		Manager::support_h_matrix_real_derivative(lpi,C.sigma,Rij,this->man_matrix4d_h_real_derivative_ws,this->man_matrix4d_h_real_derivative_out);
		factor = 0.5 * lpi->lp_charge * C.AtomList[j]->charge;

		LPC_H_Real_Derivative[i][j][0][0] -= factor * this->man_matrix4d_h_real_derivative_out[0];	
		LPC_H_Real_Derivative[i][j][0][1] -= factor * this->man_matrix4d_h_real_derivative_out[1];	
		LPC_H_Real_Derivative[i][j][0][2] -= factor * this->man_matrix4d_h_real_derivative_out[2];	// pair geometric derivative has to be considered later

		// derivative update
		this->DerivativeH[0] = -factor * this->man_matrix4d_h_real_derivative_out[0];
		this->DerivativeH[1] = -factor * this->man_matrix4d_h_real_derivative_out[1];
		this->DerivativeH[2] = -factor * this->man_matrix4d_h_real_derivative_out[2];
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;

		/* 2-1 LP(i) CORE <----- LP(j) LP */
		Rij = C.AtomList[i]->cart - ( C.AtomList[j]->cart + TransVector );
		// Derivative Evaluation ... displacement on (i)
		Manager::support_h_matrix_real_derivative(lpj,C.sigma,Rij,this->man_matrix4d_h_real_derivative_ws,this->man_matrix4d_h_real_derivative_out);
		factor = 0.5 * lpj->lp_charge * C.AtomList[i]->charge;

		LPC_H_Real_Derivative[j][i][0][0] -= factor * this->man_matrix4d_h_real_derivative_out[0];	
		LPC_H_Real_Derivative[j][i][0][1] -= factor * this->man_matrix4d_h_real_derivative_out[1];	
		LPC_H_Real_Derivative[j][i][0][2] -= factor * this->man_matrix4d_h_real_derivative_out[2];	

		// derivative update
		this->DerivativeH[0] = -factor * this->man_matrix4d_h_real_derivative_out[0];
		this->DerivativeH[1] = -factor * this->man_matrix4d_h_real_derivative_out[1];
		this->DerivativeH[2] = -factor * this->man_matrix4d_h_real_derivative_out[2];
		lpj->GetEvecGS(lpj_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[j]->cart_gd += D;
		C.AtomList[i]->cart_gd -= D;

		////	////	////	////	////	////	////	////	////	////	////	////	////	////

		/* 3-1 LP(i) LP -----> LP(j) LP ... Monopole */
		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;
		// Derivative Evaluation ... displacement on (j)
		Manager::support_h_matrix_real_derivative(lpi,C.sigma,Rij,this->man_matrix4d_h_real_derivative_ws,this->man_matrix4d_h_real_derivative_out);
		factor = 0.5 * lpi->lp_charge * lpj->lp_charge;

		LPLP_H_Real_Derivative[i][j][0] -= factor * this->man_matrix4d_h_real_derivative_out[0];
		LPLP_H_Real_Derivative[i][j][1] -= factor * this->man_matrix4d_h_real_derivative_out[1];
		LPLP_H_Real_Derivative[i][j][2] -= factor * this->man_matrix4d_h_real_derivative_out[2];

		// derivative update
		this->DerivativeH[0] = -factor * this->man_matrix4d_h_real_derivative_out[0];
		this->DerivativeH[1] = -factor * this->man_matrix4d_h_real_derivative_out[1];
		this->DerivativeH[2] = -factor * this->man_matrix4d_h_real_derivative_out[2];
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;

		/* 4 LP(i) ------> LP(j) Dipole */
		Rij = ( C.AtomList[j]->cart + TransVector ) - C.AtomList[i]->cart;
		// Derivative Evaluation ... displacement on (j)
		Manager::support_h_matrix_real_derivative2(lpi,C.sigma,Rij,man_matrix4d_h_real_derivative2_ws,man_matrix4d_h_real_derivative2_out);	// *_out[0-5] .. xx,xy,xz,yy,yz,zz
		factor = 0.5 * lpi->lp_charge * lpj->lp_charge;

		lpi->GetEvecGS(lpi_cf);
		lpj->GetEvecGS(lpj_cf);

		// For (i) 
		LPLP_H_Real_Derivative[i][j][0] -= ((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[0]*(2.*lpj_cf[0]*lpj_cf[1]) 	// 0 xx
											      + man_matrix4d_h_real_derivative2_out[1]*(2.*lpj_cf[0]*lpj_cf[2])		// 1 xy
											      + man_matrix4d_h_real_derivative2_out[2]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 2 xz
	
		LPLP_H_Real_Derivative[i][j][1] -= ((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[1]*(2.*lpj_cf[0]*lpj_cf[1])		// 1 yx (xy) 
											      + man_matrix4d_h_real_derivative2_out[3]*(2.*lpj_cf[0]*lpj_cf[2])		// 3 yy
											      + man_matrix4d_h_real_derivative2_out[4]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 4 yz

		LPLP_H_Real_Derivative[i][j][2] -= ((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[2]*(2.*lpj_cf[0]*lpj_cf[1]) 	// 2 zx (xz)
											      + man_matrix4d_h_real_derivative2_out[4]*(2.*lpj_cf[0]*lpj_cf[2])		// 4 zy (yz)
											      + man_matrix4d_h_real_derivative2_out[5]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 5 zz 

		// derivative update
		this->DerivativeH[0] = -((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[0]*(2.*lpj_cf[0]*lpj_cf[1]) 		// 0 xx
									           + man_matrix4d_h_real_derivative2_out[1]*(2.*lpj_cf[0]*lpj_cf[2])		// 1 xy
									           + man_matrix4d_h_real_derivative2_out[2]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 2 xz
		this->DerivativeH[1] = -((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[1]*(2.*lpj_cf[0]*lpj_cf[1])		// 1 yx (xy) 
										   + man_matrix4d_h_real_derivative2_out[3]*(2.*lpj_cf[0]*lpj_cf[2])		// 3 yy
										   + man_matrix4d_h_real_derivative2_out[4]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 4 yz
		this->DerivativeH[2] = -((factor*lpj->lp_real_position_integral) * ( man_matrix4d_h_real_derivative2_out[2]*(2.*lpj_cf[0]*lpj_cf[1]) 		// 2 zx (xz)
									           + man_matrix4d_h_real_derivative2_out[4]*(2.*lpj_cf[0]*lpj_cf[2])		// 4 zy (yz)
									           + man_matrix4d_h_real_derivative2_out[5]*(2.*lpj_cf[0]*lpj_cf[3]) ));	// 5 zz 
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;
		// * Correction by Evec Derivatives ... Eigen::Vector4d LP_Evec_Derivative[MX_C][MX_C][2];
		// will be considered in other function ... 
	}
	return;
}

void Manager::CoulombLonePairDerivativeReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {	
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" ) 
        {
		// Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" ) 
        {
		// Shell Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;

		// Shell Shell - LonePair Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
        
	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" ) 
        {
		// LonePair Core - Shell Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;
	
		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
		
		// LonePair Core - Shell Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact*Rij;
        }

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )    // Handling Core - Core (i.e., charge charge interaction);
        {	
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.AtomList[i]->cart_gd += intact*Rij;
		C.AtomList[j]->cart_gd -= intact*Rij;
        }       
	Manager::set_h_matrix_real_derivative( C, i, j, TransVector );

}

void Manager::CoulombLonePairDerivativeSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double factor, self;
	double lpi_cf[4];
	Eigen::Vector3d D; D.setZero();

	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" ) 
        {
		// 1. Core-Core self -> 0
		// 2. LP-LP self -> 0

		// * LP-Core self != 0
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
		Qi = lpi->lp_charge;
		Qj = C.AtomList[i]->charge;
		factor = -(0.5*Qi*Qj)*2.;
				
		//Eigen::Matrix4d LPLP_H_Self_Derivative[MX_C][3];
		//lp_self_grad = this->man_lp_matrix_h.reci_self_integral_sx_grad_x(lp->lp_r,lp->lp_r_s_function,lp->lp_r_p_function,sig);
		self = factor * this->man_lp_matrix_h.reci_self_integral_sx_grad_x(lpi->lp_r,lpi->lp_r_s_function,lpi->lp_r_p_function,C.sigma);	// unit readily ... eV/Angs .. '-' sign readily on factor
		this->LP_H_Self_Derivative[i][0](0,1) = this->LP_H_Self_Derivative[i][0](1,0) = self;	// (0,1) sx, (1,0) xs
		this->LP_H_Self_Derivative[i][1](0,2) = this->LP_H_Self_Derivative[i][1](2,0) = self;	// (0,2) sy, (2,0) ys
		this->LP_H_Self_Derivative[i][2](0,3) = this->LP_H_Self_Derivative[i][2](3,0) = self;	// (0,3) sz, (3,0) zs	// Unit already set to eV/Angs

		// derivative update
		this->DerivativeH[0] = this->LP_H_Self_Derivative[i][0];
		this->DerivativeH[1] = this->LP_H_Self_Derivative[i][1];
		this->DerivativeH[2] = this->LP_H_Self_Derivative[i][2];
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}

std::cout << std::endl;
std::cout << "SelfD------------------------------------------------------------------\n";
printf("%20.12lf\t%20.12lf\t%20.12lf\n",D(0),D(1),D(2));
std::cout << "-----------------------------------------------------------------------\n";

		C.AtomList[i]->cart_gd += D;	// j -> i limit
		C.AtomList[j]->cart_gd -= D;	// i -> j limit
	}
}       

void Manager::set_h_matrix_reci_derivative( Cell& C, const int i, const int j, const Eigen::Vector3d& G )
{
	const std::string type_i = C.AtomList[i]->type;
	const std::string type_j = C.AtomList[j]->type;
	double lpi_cf[4], lpj_cf[4];

	double factor, intact[4];	// factor : halved leading term / intact : Cos(G.Rij), Sin(G.Rij), rests->for LPLP
	double Qi,Qj;
	const double g_sqr = G.adjoint()*G;
	Eigen::Vector3d Rij;	// Saving Real Space Rij
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        Eigen::Vector3d D; D.setZero();

	// Eigen::Matrix4d LPC_H_Reci_Derivative[MX_C][MX_C][2][3];
	if( type_i == "lone" && type_j == "core" )
	{
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);

		Qi  = lpi->lp_charge;
		Qj  = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;	
	
		factor = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;
		//intact[0] = cos(Rij.adjoint()*G); // cos{(Rj-Ri).G}
		//intact[1] = sin(Rij.adjoint()*G); // sin{(Rj-Ri).G}
		intact[0] = cos(Rij.adjoint()*G); // cos{(Ri-Rj).G}
		intact[1] = sin(Rij.adjoint()*G); // sin{(Ri-Rj).G}

		Manager::support_h_matrix_reci(lpi,G,this->man_matrix4d_h_reci_ws,this->man_matrix4d_h_reci_out); // man_matrix4d_h_reci_ws/out [0..1] : 0 cosine / 1 sine ... return unit dimensionless

		// grad taken for (j)
		LPC_H_Reci_Derivative[i][j][0][0] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		LPC_H_Reci_Derivative[i][j][0][1] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		LPC_H_Reci_Derivative[i][j][0][2] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));

		// derivative update
		this->DerivativeH[0] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		this->DerivativeH[1] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		this->DerivativeH[2] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;
	}

	if( type_i == "core" && type_j == "lone" )
	{
		LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);
		
		Qi  = C.AtomList[i]->charge;
		Qj  = lpj->lp_charge;
		//Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;	// Rj(LP-Core) -----> Ri(MM-Core) ... i.e., ( Ri - Rj )
		Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;	

		factor = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;
		//intact[0] = cos(Rij.adjoint()*G); // cos{(Ri-Rj).G}
		//intact[1] = sin(Rij.adjoint()*G); // sin{(Ri-Rj).G}
		intact[0] = cos(Rij.adjoint()*G); // cos{(Rj-Ri).G}
		intact[1] = sin(Rij.adjoint()*G); // sin{(Rj-Ri).G}

		Manager::support_h_matrix_reci(lpj,G,this->man_matrix4d_h_reci_ws,this->man_matrix4d_h_reci_out); // man_matrix4d_h_reci_ws/out [0..1] : 0 cosine / 1 sine ... return unit dimensionless

		// grad taken for (i)
		LPC_H_Reci_Derivative[j][i][0][0] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		LPC_H_Reci_Derivative[j][i][0][1] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		LPC_H_Reci_Derivative[j][i][0][2] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));

		// derivative update
		this->DerivativeH[0] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		this->DerivativeH[1] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		this->DerivativeH[2] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));
		lpj->GetEvecGS(lpj_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[j]->cart_gd += D;
		C.AtomList[i]->cart_gd -= D;
	}

	if( type_i == "lone" && type_j == "shel" )
	{	// NotImplemented
	}

	if( type_i == "shel" && type_j == "lone" )
	{	//NotImplemented
	}

	if( type_i == "lone" && type_j == "lone" )
	{
		LonePair* lpi = static_cast<LonePair*>(C.AtomList[i]);
		LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);

		// LPi (LP) -----> LPj (Core)
		Qi  = lpi->lp_charge;
		Qj  = C.AtomList[j]->charge;
		//Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		factor = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;
		//intact[0] = cos(Rij.adjoint()*G); // cos{(Rj-Ri).G}
		//intact[1] = sin(Rij.adjoint()*G); // sin{(Rj-Ri).G}
		intact[0] = cos(Rij.adjoint()*G); // cos{(Ri-Rj).G}
		intact[1] = sin(Rij.adjoint()*G); // sin{(Ri-Rj).G}

		Manager::support_h_matrix_reci(lpi,G,this->man_matrix4d_h_reci_ws,this->man_matrix4d_h_reci_out); // man_matrix4d_h_reci_ws/out [0..1] : 0 cosine / 1 sine ... return unit dimensionless

		// grad taken for (j)
		LPC_H_Reci_Derivative[i][j][0][0] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		LPC_H_Reci_Derivative[i][j][0][1] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		LPC_H_Reci_Derivative[i][j][0][2] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));

		// derivative update
		this->DerivativeH[0] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		this->DerivativeH[1] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		this->DerivativeH[2] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;

		// LPi (Core) <----- LPj (LP)
		Qi  = C.AtomList[i]->charge;
		Qj  = lpj->lp_charge;
		//Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;
		Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;

		factor = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;
		//intact[0] = cos(Rij.adjoint()*G); // cos{(Ri-Rj).G}
		//intact[1] = sin(Rij.adjoint()*G); // sin{(Ri-Rj).G}
		intact[0] = cos(Rij.adjoint()*G); // cos{(Rj-Ri).G}
		intact[1] = sin(Rij.adjoint()*G); // sin{(Rj-Ri).G}

		Manager::support_h_matrix_reci(lpj,G,this->man_matrix4d_h_reci_ws,this->man_matrix4d_h_reci_out); // man_matrix4d_h_reci_ws/out [0..1] : 0 cosine / 1 sine ... return unit dimensionless

		// grad taken for (i) 
		LPC_H_Reci_Derivative[j][i][0][0] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		LPC_H_Reci_Derivative[j][i][0][1] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		LPC_H_Reci_Derivative[j][i][0][2] -= (factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));

		// derivative update
		this->DerivativeH[0] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(0) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(0) ));
		this->DerivativeH[1] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(1) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(1) ));
		this->DerivativeH[2] = -(factor*( this->man_matrix4d_h_reci_out[0]*(intact[1])*G(2) - this->man_matrix4d_h_reci_out[1]*-intact[0]*G(2) ));
		lpj->GetEvecGS(lpj_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpj_cf[u]*lpj_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[j]->cart_gd += D;
		C.AtomList[i]->cart_gd -= D;

		// LPi -----> LPj	// Ref? lpj->GetEvecGS( lpj_cf );
		Qi  = lpi->lp_charge;
		Qj  = lpj->lp_charge;
		//Rij = C.AtomList[j]->cart - C.AtomList[i]->cart;	// Rj - Ri
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;	// Rj - Ri

		lpj->GetEvecGS( lpj_cf ); // get cf of (j)

		factor = C.TO_EV * (2.*M_PI/C.volume)*(Qi*Qj)*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr;
		//intact[0] = cos(Rij.adjoint()*G); // cos{(Rj-Ri).G}
		//intact[1] = sin(Rij.adjoint()*G); // sin{(Rj-Ri).G}
		intact[0] = cos(Rij.adjoint()*G); // cos{(Ri-Rj).G}
		intact[1] = sin(Rij.adjoint()*G); // sin{(Ri-Rj).G}

		Manager::support_h_matrix_reci(lpi,G,this->man_matrix4d_h_reci_ws,this->man_matrix4d_h_reci_out); // man_matrix4d_h_reci_ws/out [0..1] : 0 cosine / 1 sine ... return unit dimensionless

		// grad taken for (j)
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	for(int l=0;l<4;l++)
				{	for(int s=0;s<4;s++)
					{
		  this->LPLP_H_Reci_Derivative[i][j][0](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(0)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(0)*this->man_matrix4d_h_reci_out[1](u,v))
											    +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(0)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(0)*this->man_matrix4d_h_reci_out[0](u,v))));

		  this->LPLP_H_Reci_Derivative[i][j][1](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(1)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(1)*this->man_matrix4d_h_reci_out[1](u,v))
											    +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(1)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(1)*this->man_matrix4d_h_reci_out[0](u,v))));	

		  this->LPLP_H_Reci_Derivative[i][j][2](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(2)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(2)*this->man_matrix4d_h_reci_out[1](u,v))
											    +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(2)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(2)*this->man_matrix4d_h_reci_out[0](u,v))));	
					}
				}
			}
		}

		this->DerivativeH[0].setZero(); this->DerivativeH[1].setZero(); this->DerivativeH[2].setZero();
		// derivative update
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	for(int l=0;l<4;l++)
				{	for(int s=0;s<4;s++)
					{
				  this->DerivativeH[0](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(0)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(0)*this->man_matrix4d_h_reci_out[1](u,v))
											   +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(0)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(0)*this->man_matrix4d_h_reci_out[0](u,v))));

				  this->DerivativeH[1](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(1)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(1)*this->man_matrix4d_h_reci_out[1](u,v))
											   +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(1)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(1)*this->man_matrix4d_h_reci_out[0](u,v))));	

				  this->DerivativeH[2](u,v) -= factor*(lpj_cf[l]*lpj_cf[s]*(this->man_matrix4d_h_reci_out[0](l,s)*(intact[1]*G(2)*this->man_matrix4d_h_reci_out[0](u,v) - (-intact[0])*G(2)*this->man_matrix4d_h_reci_out[1](u,v))
											   +this->man_matrix4d_h_reci_out[1](l,s)*(intact[1]*G(2)*this->man_matrix4d_h_reci_out[1](u,v) + (-intact[0])*G(2)*this->man_matrix4d_h_reci_out[0](u,v))));	
					}
				}
			}
		}
		lpi->GetEvecGS(lpi_cf);
		for(int u=0;u<4;u++)
		{	for(int v=0;v<4;v++)
			{	D(0) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[0](u,v);
				D(1) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[1](u,v);
				D(2) += lpi_cf[u]*lpi_cf[v]*this->DerivativeH[2](u,v);
			}
		}
		C.AtomList[i]->cart_gd += D;
		C.AtomList[j]->cart_gd -= D;
		/* Description add!  Feb 10 2023

		H_LPLP_Reci[a][b](u,v) += f * cb_l * cb_s * ( CosInt_b(l,s) * ( cos(G.Rab) * CosInt_a(u,v) - sin(G.Rab) * SinInt_a(u,v) )
							    + SinInt_b(l,s) * ( cos(G.Rab) * SinInt_a(u,v) + sin(G.Rab) * CosInt_a(u,v) ) );
 			
		Rab = Rb - Ra; derivative is taken w.r.t. 'b' but want to keep for 'a', therefore sign is inverted..

		e.g. 
			dxb cos(G.Rab) = -sin(G.Rab) * Gx , dxb sin(G.Rab) = cos(G.Rab) * Gx
			dyb cos(G.Rab) = -sin(G.Rab) * Gy , dyb sin(G.Rab) = cos(G.Rab) * Gy
			dzb cos(G.Rab) = -sin(G.Rab) * Gz , dzb sin(G.Rab) = cos(G.Rab) * Gz

		dxHab[a][b](u,v) -= f * cb_l * cb_s * ( CosInt_b(l,s) * ( -sin(G.Rab)*Gx * CosInt_a(u,v) - cos(G.Rab)*Gx * SinInt_a(u,v) )
						      + SinInt_b(l,s) * ( -sin(G.Rab)*Gx * SinInt_a(u,v) + cos(G.Rab)*Gx * CosInt_a(u,v) )

		dyHab[a][b](u,v) -= f * cb_l * cb_s * ( CosInt_b(l,s) * ( -sin(G.Rab)*Gy * CosInt_a(u,v) - cos(G.Rab)*Gy * SinInt_a(u,v) )
						      + SinInt_b(l,s) * ( -sin(G.Rab)*Gy * SinInt_a(u,v) + cos(G.Rab)*Gy * CosInt_a(u,v) )

		dzHab[a][b](u,v) -= f * cb_l * cb_s * ( CosInt_b(l,s) * ( -sin(G.Rab)*Gz * CosInt_a(u,v) - cos(G.Rab)*Gz * SinInt_a(u,v) )
						      + SinInt_b(l,s) * ( -sin(G.Rab)*Gz * SinInt_a(u,v) + cos(G.Rab)*Gz * CosInt_a(u,v) )
 		*/
	}

	return;
}

void Manager::CoulombLonePairDerivativeReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector /* G */)
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;
	double intact;
        
        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {	
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" ) 
        {	
		// Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" ) 
        {
		// Shell Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// Shell Shell - LonePair Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;
	
		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		static_cast<Shell*>(C.AtomList[i])->shel_cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }       
        
	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" ) 
        {
		// LonePair Core - Shell Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;

		// LonePair Core - Shell Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		static_cast<Shell*>(C.AtomList[j])->shel_cart_gd -= intact * TransVector;
        }       

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )    // Handling Core - Core (i.e., charge charge interaction);
        {	
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact = C.TO_EV*((2.*M_PI)/C.volume)*Qi*Qj*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);

		C.AtomList[i]->cart_gd += intact * TransVector;
		C.AtomList[j]->cart_gd -= intact * TransVector;
        }
	Manager::set_h_matrix_reci_derivative( C, i, j, TransVector );	// TransVector = G
}       



/* Atom.hpp Reference
 *
 * int GetGSIndex() { return this->lp_gs_index; }
 * double GetEval( const int i ){ return this->lp_eigensolver.eigenvalues()[i].real(); }                           // Get (i) Eval
 * double GetEvec( const int i, const int j ){ return this->lp_eigensolver.eigenvectors()(i,j).real(); }           // Get (i,j) Evec ---> 'j' is a state
 * void GetEvecGS( double (&v)[4] )
 */

void Manager::grad_evec_cart_solver_support( Cell& C, const Eigen::Matrix4d (&dh_matrix)[3], const int i /* differentiate with */, const int j /* tar */, const int mode )
{
	LonePair* lpj = static_cast<LonePair*>(C.AtomList[j]);
	const int gs = lpj->GetGSIndex();	//static_cast<LonePair*>(C.AtomList[j])->GetGSIndex();
	double de[3] = {0.,0.,0.};
	double dc[3] = {0.,0.,0.};
	// eval : lpj->GetEval(i);
	// evec : lpj->GetEvec(i,j); where (j): n, state, and (i): s(0), px(1), py(2), pz(3)

	/* parameter 'mode'

	*/ 

	// Calculating dEj/dxi ... (j) .eq. alpha
	
	for(int u=0;u<4;u++)
	{	for(int v=0;v<4;v++)
		{
			de[0] += lpj->GetEvec(u,gs)*lpj->GetEvec(v,gs)*dh_matrix[0](u,v);
			de[1] += lpj->GetEvec(u,gs)*lpj->GetEvec(v,gs)*dh_matrix[1](u,v);
			de[2] += lpj->GetEvec(u,gs)*lpj->GetEvec(v,gs)*dh_matrix[2](u,v);
		}
	}

	for(int r=0;r<4;r++) // loop: s px py pz of (j .eq. alpha) subscript .eq. gamma(r)
	{
		for(int m=0;m<4;m++) // loop: states
		{
			if( m != gs )	// m != gs (n) .eq. ground state
			{
				for(int u=0;u<4;u++)
				{	for(int v=0;v<4;v++)
					{	// CPHF equation
						dc[0] += ( lpj->GetEvec(u,m) * lpj->GetEvec(r,m) / ( lpj->GetEval(m) - lpj->GetEval(gs) ) ) * ( lpj->GetEvec(v,gs) * de[0] * kDelta(u,v) - lpj->GetEvec(v,gs) * dh_matrix[0](u,v) );
						dc[1] += ( lpj->GetEvec(u,m) * lpj->GetEvec(r,m) / ( lpj->GetEval(m) - lpj->GetEval(gs) ) ) * ( lpj->GetEvec(v,gs) * de[1] * kDelta(u,v) - lpj->GetEvec(v,gs) * dh_matrix[1](u,v) );
						dc[2] += ( lpj->GetEvec(u,m) * lpj->GetEvec(r,m) / ( lpj->GetEval(m) - lpj->GetEval(gs) ) ) * ( lpj->GetEvec(v,gs) * de[2] * kDelta(u,v) - lpj->GetEvec(v,gs) * dh_matrix[2](u,v) );

					}// v
				}// y
			}// cond: m!=gs
		}// m
	
		if( mode == 0 ) // if 'i' is LP
		{
			this->grad_evec_lp_aux[i][j][0](r) = dc[0]; // deruvatuve of cj_r w.r.t xi
			this->grad_evec_lp_aux[i][j][1](r) = dc[1];
			this->grad_evec_lp_aux[i][j][2](r) = dc[2];
		}
		if( mode == 1 ) // if 'i' is mm core
		{
			this->grad_evec_mm_aux[i][j][0][0](r) = dc[0];
			this->grad_evec_mm_aux[i][j][0][1](r) = dc[1];
			this->grad_evec_mm_aux[i][j][0][2](r) = dc[2];
		}

		dc[0] = dc[1] = dc[2] = 0.; // reset

	}// end r

	return;
}

void Manager::grad_evec_cart_solver( Cell& C )
{
	const int cycmx = 100;
	double ssqr = 0.;

	LonePair* lpk;
	double evk[4];

	Eigen::Matrix4d dh_matrix_tmp[3];	// [3] ... for x, y and z
	dh_matrix_tmp[0].setZero(); dh_matrix_tmp[1].setZero(); dh_matrix_tmp[2].setZero();

	// in Manager.hpp
	// Eigen::Vector4d grad_evec_aux[MX_C][MX_C][2][3];
	// Eigen::Vector4d grad_evec[MX_C][MX_C][2][3];    // [i][j][2][3]: Rij(=Rj-Ri) Grad(i) // [2] : shell / core // [3] : x, y and z

	for(int i=0;i<C.NumberOfAtoms;i++)
	{	for(int j=0;j<C.NumberOfAtoms;j++)
		{	for(int k=0;k<3;k++)
			{								// dcj^()/dki
				this->grad_evec_lp_aux[i][j][k].setZero();	this->grad_evec_lp[i][j][k].setZero();
				this->grad_evec_mm_aux[i][j][0][k].setZero();	this->grad_evec_mm[i][j][0][k].setZero();	// [0] core
				this->grad_evec_mm_aux[i][j][1][k].setZero();	this->grad_evec_mm[i][j][1][k].setZero();	// [1] shel
					// dcj^()/dki_shell
			}
		}
	}

	/*
		Using Jacobi-Iteration Method
	*/
	for(int cyc=0;cyc<cycmx;cyc++)
	{
		/* Convention (in Thesis)
 		   
 		   j -> alpha
		   k -> beta

		   i -> differentiate with (i)th something ... in general 'Cartesian Coordinate'
		*/
	
		// Step 1. versus LP centres
		for(int i=0;i<C.NumberOfAtoms;i++) // differentiate with (i)
		{	
			if( C.AtomList[i]->type == "lone" )
			{	
				for(int j=0;j<C.NumberOfAtoms;j++) // (j) .eq. alpha
				{	
					if( C.AtomList[j]->type == "lone" )
					{	
						if( i == j ) // here i == j means all MM must be included!! - see the master eqns in phD thesis
						{
							// versus MM ions
							for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. A
							{
								if( C.AtomList[k]->type == "core" )
								{
									dh_matrix_tmp[0] += this->LPC_H_Real_Derivative[i][k][0][0];
									dh_matrix_tmp[1] += this->LPC_H_Real_Derivative[i][k][0][1];
									dh_matrix_tmp[2] += this->LPC_H_Real_Derivative[i][k][0][2];

									dh_matrix_tmp[0] += this->LPC_H_Reci_Derivative[i][k][0][0];
									dh_matrix_tmp[1] += this->LPC_H_Reci_Derivative[i][k][0][1];
									dh_matrix_tmp[2] += this->LPC_H_Reci_Derivative[i][k][0][2];
								}

								if( C.AtomList[k]->type == "shel" )
								{	// NotImplemented
								}
							}

							for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. beta // note that i = j
							{
								if( C.AtomList[k]->type == "lone" )
								{
									if( k != j )
									{
										lpk = static_cast<LonePair*>(C.AtomList[k]);
										lpk->GetEvecGS(evk);
										// versus LP (k) core real
										dh_matrix_tmp[0] += this->LPC_H_Real_Derivative[i][k][0][0];
										dh_matrix_tmp[1] += this->LPC_H_Real_Derivative[i][k][0][1];
										dh_matrix_tmp[2] += this->LPC_H_Real_Derivative[i][k][0][2];
										// versus LP (k) core reci
										dh_matrix_tmp[0] += this->LPC_H_Reci_Derivative[i][k][0][0];
										dh_matrix_tmp[1] += this->LPC_H_Reci_Derivative[i][k][0][1];
										dh_matrix_tmp[2] += this->LPC_H_Reci_Derivative[i][k][0][2];
		
										// versus LP (k) monopole and dipole real
										dh_matrix_tmp[0] += this->LPLP_H_Real_Derivative[i][k][0];
										dh_matrix_tmp[1] += this->LPLP_H_Real_Derivative[i][k][1];
										dh_matrix_tmp[2] += this->LPLP_H_Real_Derivative[i][k][2];
										// versus LP (k)
										dh_matrix_tmp[0] += this->LPLP_H_Reci_Derivative[i][k][0];
										dh_matrix_tmp[1] += this->LPLP_H_Reci_Derivative[i][k][1];
										dh_matrix_tmp[2] += this->LPLP_H_Reci_Derivative[i][k][2];

										// Contribution by EvecDerivatives

										// Real			           dck(0;s)/dxi        ck(1;px)
										dh_matrix_tmp[0] += (  2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Real_Aux[i][k][0]
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Real_Aux[i][k][1]
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Real_Aux[i][k][2] );

										dh_matrix_tmp[1] += (  2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Real_Aux[i][k][0]
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Real_Aux[i][k][1]
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Real_Aux[i][k][2] );

										dh_matrix_tmp[2] += (  2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Real_Aux[i][k][0]
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Real_Aux[i][k][1]
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Real_Aux[i][k][2] );

										// Reci
										dh_matrix_tmp[0] += (     (this->grad_evec_lp[i][k][0](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][0](0))*this->LPLP_H_Reci_Aux[i][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[i][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][0](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[i][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][0](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][0](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][0](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][9] );	//  zz

										dh_matrix_tmp[1] += (     (this->grad_evec_lp[i][k][1](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][1](0))*this->LPLP_H_Reci_Aux[i][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[i][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][1](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[i][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][1](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][1](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][1](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][9] );	//  zz

										dh_matrix_tmp[2] += (     (this->grad_evec_lp[i][k][2](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][2](0))*this->LPLP_H_Reci_Aux[i][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[i][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][2](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[i][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][2](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][2](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][2](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][9] );	//  zz
									} // end : if( k != j )

									if( k == j ) // self interaction ... LPcore - LP
									{
										dh_matrix_tmp[0] += this->LP_H_Self_Derivative[i][0];
										dh_matrix_tmp[1] += this->LP_H_Self_Derivative[i][1];
										dh_matrix_tmp[2] += this->LP_H_Self_Derivative[i][2];
									} 

								}// end : if( C.AtomList[k]->type == "lone" )

							}// end : for(int k=0;k<C.NumberOfAtoms;k++) // versus LP

							// DO_SOMETHING ... CPHF to update evec derivatives - not Implemented
							Manager::grad_evec_cart_solver_support( C, dh_matrix_tmp, i, j, 0 );
							dh_matrix_tmp[0].setZero(); dh_matrix_tmp[1].setZero(); dh_matrix_tmp[2].setZero();

						}// end : if( i == j )
						
						if( i != j ) // defferentiate (j)-LP w.r.t. (i)-LP dHjk/dxi but i != j; therefore, MM terms vanish, also i!=j -> no self derivative contribution
						{
							for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. beta // (j) .eq. alpha
							{
								if( C.AtomList[k]->type == "lone" )
								{
									if( k != j )
									{
										lpk = static_cast<LonePair*>(C.AtomList[k]);
										lpk->GetEvecGS(evk);

										if( k == i )	// differentiate with 'i', therfore only applied when k == i
										{
											// versus LP (i) core Real - sign inverted
											dh_matrix_tmp[0] += -this->LPC_H_Real_Derivative[j][i][0][0];
											dh_matrix_tmp[1] += -this->LPC_H_Real_Derivative[j][i][0][1];
											dh_matrix_tmp[2] += -this->LPC_H_Real_Derivative[j][i][0][2];
											// versus LP (i) core Reci - sign inverted
											dh_matrix_tmp[0] += -this->LPC_H_Reci_Derivative[j][i][0][0];
											dh_matrix_tmp[1] += -this->LPC_H_Reci_Derivative[j][i][0][1];
											dh_matrix_tmp[2] += -this->LPC_H_Reci_Derivative[j][i][0][2];
											// versus LP (i) monopole and dipole real - sign inverted
											dh_matrix_tmp[0] += -this->LPLP_H_Real_Derivative[j][i][0];			
											dh_matrix_tmp[1] += -this->LPLP_H_Real_Derivative[j][i][1];			
											dh_matrix_tmp[2] += -this->LPLP_H_Real_Derivative[j][i][2];			
											// versus LP (i) - sign inverted
											dh_matrix_tmp[0] += -this->LPLP_H_Reci_Derivative[j][i][0];			
											dh_matrix_tmp[1] += -this->LPLP_H_Reci_Derivative[j][i][1];			
											dh_matrix_tmp[2] += -this->LPLP_H_Reci_Derivative[j][i][2];			
										} // end : if( k == i )

										// Contribution by Evec Derivatives
										// Real			                dck(0:s)/dxi * ck(1;px)
										dh_matrix_tmp[0] += (  2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Real_Aux[j][k][0]
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Real_Aux[j][k][1]
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Real_Aux[j][k][2] );

										dh_matrix_tmp[1] += (  2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Real_Aux[j][k][0]
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Real_Aux[j][k][1]
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Real_Aux[j][k][2] );

										dh_matrix_tmp[2] += (  2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Real_Aux[j][k][0]
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Real_Aux[j][k][1]
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Real_Aux[j][k][2] );
									
										// Reci
										dh_matrix_tmp[0] += (     (this->grad_evec_lp[i][k][0](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][0](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][0](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][0](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][0](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][0](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz

										dh_matrix_tmp[1] += (     (this->grad_evec_lp[i][k][1](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][1](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][1](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][1](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][1](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][1](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz

										dh_matrix_tmp[2] += (     (this->grad_evec_lp[i][k][2](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][2](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
												     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
												     +    (this->grad_evec_lp[i][k][2](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
												     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
												     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
												     +    (this->grad_evec_lp[i][k][2](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
												     + 2.*(this->grad_evec_lp[i][k][2](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
												     +    (this->grad_evec_lp[i][k][2](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz
									}// end : if( k != j )

								}// end : if( C.AtomList[k]->type == "lone" )

							}// end : for(int k=0;k<C.NumberOfAtoms;k++) // versus other LP Centres

							// DO SOMETHING ... CPHF to update evec derivatives - not Implemented
							Manager::grad_evec_cart_solver_support( C, dh_matrix_tmp, i, j, 0 );
							dh_matrix_tmp[0].setZero(); dh_matrix_tmp[1].setZero(); dh_matrix_tmp[2].setZero();

						}// end : if( i != j )

					}// end : if( C.AtomList[j]->type == "lone" ) // if (j) is LP

				}// end : for(int j=0;j<C.NumberOfAtoms;j++) ; (j) --- alpha (if using thesis convention)

			}// end : if( C.AtomList[i]->type == "lone" )     // if (i) is LP

		}// end : for(int i=0;i<C.NumberOfAtoms;i++) // Finalise Step 1.


		// Step 2. versus MM Centres ... curret version ~ 13 Feb 2023 only 'cores'
		for(int i=0;i<C.NumberOfAtoms;i++) // differentiate with the (i)th MM core
		{
			if( C.AtomList[i]->type == "core" )
			{
				for(int j=0;j<C.NumberOfAtoms;j++) // (j) .eq. alpha
				{
					if( C.AtomList[j]->type == "lone" )
					{
						for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. A
						{
							if( C.AtomList[k]->type == "core" )
							{
								if( k == i )
								{	// Real
									dh_matrix_tmp[0] += -this->LPC_H_Real_Derivative[j][i][0][0];	 // note i == k	 where (i) is core
									dh_matrix_tmp[1] += -this->LPC_H_Real_Derivative[j][i][0][1];
									dh_matrix_tmp[2] += -this->LPC_H_Real_Derivative[j][i][0][2];
									// Reci
									dh_matrix_tmp[0] += -this->LPC_H_Reci_Derivative[j][i][0][0];
									dh_matrix_tmp[1] += -this->LPC_H_Reci_Derivative[j][i][0][1];
									dh_matrix_tmp[2] += -this->LPC_H_Reci_Derivative[j][i][0][2];
								}// end : if( k == i )

							}// end : if( C.AtomList[k] == "core" )

						}// end : for(int k=0;k<C.NumberOfAtoms;k++)
						
						for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. beta
						{
							if( C.AtomList[k]->type == "lone" )
							{
								if( k != j )
								{
									lpk = static_cast<LonePair*>(C.AtomList[k]);
									lpk->GetEvecGS(evk);

									// Contribution by Evec Derivatives
									// Real			                dck(0:s)/dxi * ck(1;px)
									dh_matrix_tmp[0] += (  2.*(this->grad_evec_mm[i][k][0][0](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Real_Aux[j][k][0]
											     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Real_Aux[j][k][1]
											     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Real_Aux[j][k][2] );
															  //^-core [0]	i.e., (i) is core		      ^-core [0]
									dh_matrix_tmp[1] += (  2.*(this->grad_evec_mm[i][k][0][1](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Real_Aux[j][k][0]
											     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Real_Aux[j][k][1]
											     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Real_Aux[j][k][2] );

									dh_matrix_tmp[2] += (  2.*(this->grad_evec_mm[i][k][0][2](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Real_Aux[j][k][0]
											     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Real_Aux[j][k][1]
											     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Real_Aux[j][k][2] );
									
									// Reci
									dh_matrix_tmp[0] += (     (this->grad_evec_mm[i][k][0][0](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][0](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
											     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
											     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
											     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
											     +    (this->grad_evec_mm[i][k][0][0](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
											     + 2.*(this->grad_evec_mm[i][k][0][0](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
											     + 2.*(this->grad_evec_mm[i][k][0][0](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
											     +    (this->grad_evec_mm[i][k][0][0](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
											     + 2.*(this->grad_evec_mm[i][k][0][0](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
											     +    (this->grad_evec_mm[i][k][0][0](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz

									dh_matrix_tmp[1] += (     (this->grad_evec_mm[i][k][0][1](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][1](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
											     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
											     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
											     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
											     +    (this->grad_evec_mm[i][k][0][1](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
											     + 2.*(this->grad_evec_mm[i][k][0][1](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
											     + 2.*(this->grad_evec_mm[i][k][0][1](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
											     +    (this->grad_evec_mm[i][k][0][1](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
											     + 2.*(this->grad_evec_mm[i][k][0][1](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
											     +    (this->grad_evec_mm[i][k][0][1](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz

									dh_matrix_tmp[2] += (     (this->grad_evec_mm[i][k][0][2](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][2](0))*this->LPLP_H_Reci_Aux[j][k][0]	//  ss
											     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Reci_Aux[j][k][1]	// 2sx
											     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][2]	// 2sy
											     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][3]	// 2sz
											     +    (this->grad_evec_mm[i][k][0][2](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Reci_Aux[j][k][4]	//  xx
											     + 2.*(this->grad_evec_mm[i][k][0][2](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][5]	// 2xy
											     + 2.*(this->grad_evec_mm[i][k][0][2](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][6]	// 2xz
											     +    (this->grad_evec_mm[i][k][0][2](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][7]	//  yy
											     + 2.*(this->grad_evec_mm[i][k][0][2](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][8]	//  yz
											     +    (this->grad_evec_mm[i][k][0][2](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][9] );	//  zz

								
								}// end : if( k != j )

							}// end : if( C.AtomList[k] == "lone" )

						}// end : for(int k=0;k<C.NumberOfAtoms;k++) // (k) .eq. beta
						
						// DO SOMETHING ... CPHF to update evec derivatives - not Implemented
						Manager::grad_evec_cart_solver_support( C, dh_matrix_tmp, i, j, 1 );
						dh_matrix_tmp[0].setZero(); dh_matrix_tmp[1].setZero(); dh_matrix_tmp[2].setZero();

					}// end : if( C.AtomList[j]->type == "lone" )

				}// end : for(int j=0;j<C.NumberOfAtoms;j++) // (j) .eq. alpha

			}// end : if( C.AtomList[i]->type == "core" )

		}// end : for(int i=0;i<C.NumberOfAtoms;i++) // differentiate with (i)

		/*
 			Evec Derivative update
			
			grad_evec_lp[i][j][3](c) / grad_evec_lp_aux[i][j][3](c)

			grad_evec_mm[i][j][2][3](c) / grad_evec_mm_aux[i][j][2][3](c)
		*/

		for(int i=0;i<C.NumberOfAtoms;i++)
		{	if( C.AtomList[i]->type == "lone" )
			{	for(int j=0;j<C.NumberOfAtoms;j++)
				{	if( C.AtomList[j]->type == "lone" )
					{	for(int k=0;k<3;k++)
						{	for(int c=0;c<4;c++)
							{
								ssqr += pow(this->grad_evec_lp_aux[i][j][k](c)-this->grad_evec_lp[i][j][k](c),2.);
								this->grad_evec_lp[i][j][k](c) = this->grad_evec_lp_aux[i][j][k](c);
							}
						}
					}
				}
			}
		}

		for(int i=0;i<C.NumberOfAtoms;i++)
		{	if( C.AtomList[i]->type == "core" )
			{	for(int j=0;j<C.NumberOfAtoms;j++)
				{	if( C.AtomList[j]->type == "lone" )
					{	for(int k=0;k<3;k++)
						{	for(int c=0;c<4;c++)
							{
								ssqr += pow(this->grad_evec_mm_aux[i][j][0][k](c)-this->grad_evec_mm[i][j][0][k](c),2.);
								this->grad_evec_mm[i][j][0][k](c) = this->grad_evec_mm_aux[i][j][0][k](c);
							}
						}
					}
				}
			}
		}

		ssqr = sqrt(ssqr)/(3.*C.NumberOfAtoms);
		printf("%d%20.12lf\n",cyc+1,ssqr);

		if( ssqr < 10E-12 ){ break; }	// testing tolerance
		ssqr = 0.;

	}// end : for(cyc)
	return;
}


void Manager::LonePairDerivativeCorrection( Cell& C )
{
	// Correction by EigenVector Derivatives
	// LP, MM-Core system only 14 Feb 2023
	LonePair* lpj = nullptr;
	LonePair* lpk = nullptr;

	double evj[4], evk[4];
	double dcorr[3] = {0.,0.,0.};

	// Evec Derivative Correction LP
	for(int i=0;i<C.NumberOfAtoms;i++)
	{
		if( C.AtomList[i]->type == "lone" )
		{
			for(int j=0;j<C.NumberOfAtoms;j++)
			{	
				if( C.AtomList[j]->type == "lone" )
				{
					lpj = static_cast<LonePair*>(C.AtomList[j]);
					lpj->GetEvecGS(evj);

					for(int k=0;k<C.NumberOfAtoms;k++)
					{
						if( C.AtomList[k]->type == "lone" )
						{
							lpk = static_cast<LonePair*>(C.AtomList[k]);
							lpk->GetEvecGS(evk);

							if( k != j )
							{
								for(int u=0;u<4;u++)
								{	for(int v=0;v<4;v++)
									{
										// Real
										dcorr[0] += evj[u]*evj[v]*( 2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Real_Aux[i][k][0](u,v)
													  + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Real_Aux[i][k][1](u,v)
													  + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Real_Aux[i][k][2](u,v) );

										dcorr[1] += evj[u]*evj[v]*( 2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Real_Aux[i][k][0](u,v)
													  + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Real_Aux[i][k][1](u,v)
													  + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Real_Aux[i][k][2](u,v) );

										dcorr[2] += evj[u]*evj[v]*( 2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Real_Aux[i][k][0](u,v)
													  + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Real_Aux[i][k][1](u,v)
													  + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Real_Aux[i][k][2](u,v) );
										// Reci
										dcorr[0] += evj[u]*evj[v]*(	  (this->grad_evec_lp[i][k][0](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][0](0))*this->LPLP_H_Reci_Aux[i][k][0](u,v)	//  ss
													     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[i][k][1](u,v)	// 2sx
													     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][2](u,v)	// 2sy
													     + 2.*(this->grad_evec_lp[i][k][0](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][3](u,v)	// 2sz
													     +    (this->grad_evec_lp[i][k][0](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][0](1))*this->LPLP_H_Reci_Aux[i][k][4](u,v)	//  xx
													     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][5](u,v)	// 2xy
													     + 2.*(this->grad_evec_lp[i][k][0](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][6](u,v)	// 2xz
													     +    (this->grad_evec_lp[i][k][0](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][0](2))*this->LPLP_H_Reci_Aux[i][k][7](u,v)	//  yy
													     + 2.*(this->grad_evec_lp[i][k][0](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][8](u,v)	//  yz
													     +    (this->grad_evec_lp[i][k][0](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][0](3))*this->LPLP_H_Reci_Aux[i][k][9](u,v) );//  zz

										dcorr[1] += evj[u]*evj[v]*(       (this->grad_evec_lp[i][k][1](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][1](0))*this->LPLP_H_Reci_Aux[i][k][0](u,v)	//  ss
													     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[i][k][1](u,v)	// 2sx
													     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][2](u,v)	// 2sy
													     + 2.*(this->grad_evec_lp[i][k][1](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][3](u,v)	// 2sz
													     +    (this->grad_evec_lp[i][k][1](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][1](1))*this->LPLP_H_Reci_Aux[i][k][4](u,v)	//  xx
													     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][5](u,v)	// 2xy
													     + 2.*(this->grad_evec_lp[i][k][1](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][6](u,v)	// 2xz
													     +    (this->grad_evec_lp[i][k][1](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][1](2))*this->LPLP_H_Reci_Aux[i][k][7](u,v)	//  yy
													     + 2.*(this->grad_evec_lp[i][k][1](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][8](u,v)	//  yz
													     +    (this->grad_evec_lp[i][k][1](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][1](3))*this->LPLP_H_Reci_Aux[i][k][9](u,v) );//  zz

										dcorr[2] += evj[u]*evj[v]*(       (this->grad_evec_lp[i][k][2](0)*evk[0] + evk[0]*this->grad_evec_lp[i][k][2](0))*this->LPLP_H_Reci_Aux[i][k][0](u,v)	//  ss
													     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[1] + evk[0]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[i][k][1](u,v)	// 2sx
													     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[2] + evk[0]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][2](u,v)	// 2sy
													     + 2.*(this->grad_evec_lp[i][k][2](0)*evk[3] + evk[0]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][3](u,v)	// 2sz
													     +    (this->grad_evec_lp[i][k][2](1)*evk[1] + evk[1]*this->grad_evec_lp[i][k][2](1))*this->LPLP_H_Reci_Aux[i][k][4](u,v)	//  xx
													     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[2] + evk[1]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][5](u,v)	// 2xy
													     + 2.*(this->grad_evec_lp[i][k][2](1)*evk[3] + evk[1]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][6](u,v)	// 2xz
													     +    (this->grad_evec_lp[i][k][2](2)*evk[2] + evk[2]*this->grad_evec_lp[i][k][2](2))*this->LPLP_H_Reci_Aux[i][k][7](u,v)	//  yy
													     + 2.*(this->grad_evec_lp[i][k][2](2)*evk[3] + evk[2]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][8](u,v)	//  yz
													     +    (this->grad_evec_lp[i][k][2](3)*evk[3] + evk[3]*this->grad_evec_lp[i][k][2](3))*this->LPLP_H_Reci_Aux[i][k][9](u,v) );//  zz
									}// v
								}// u
							}// end : if( k != j )
						}// end : if( C.AtomList[k]->type == "lone" )
					}// end : for(int k=0;k<C.NumberOfAtoms;k++)
				}// end : if( C.AtomList[j]->type == "lone" )
			}// end : for(int j=0;j<C.NumberOfAtoms;j++)
		}//end : if( C.AtomList[i]->type == "lone" )

		C.AtomList[i]->cart_gd(0) += dcorr[0];
		C.AtomList[i]->cart_gd(1) += dcorr[1];
		C.AtomList[i]->cart_gd(2) += dcorr[2];

		//printf("Check-1\n");
		//printf("%10.4lf\t%10.4lf\t%10.4lf\n",dcorr[0],dcorr[1],dcorr[2]);
		
		dcorr[0] = dcorr[1] = dcorr[2] = 0.;
	}// end : for(int i=0;i<C.NumberOfAtoms;i++)


	// Evec Derivative Correction - MM
	for(int i=0;i<C.NumberOfAtoms;i++)
	{
		if( C.AtomList[i]->type == "core" )
		{
			for(int j=0;j<C.NumberOfAtoms;j++)
			{	
				if( C.AtomList[j]->type == "lone" )
				{
					lpj = static_cast<LonePair*>(C.AtomList[j]);
					lpj->GetEvecGS(evj);

					for(int k=0;k<C.NumberOfAtoms;k++)
					{
						if( C.AtomList[k]->type == "lone" )
						{
							lpk = static_cast<LonePair*>(C.AtomList[k]);
							lpk->GetEvecGS(evk);

							if( k != j )
							{
								for(int u=0;u<4;u++)
								{	for(int v=0;v<4;v++)
									{
									// Real
									dcorr[0] += evj[u]*evj[v] * (  2.*(this->grad_evec_mm[i][k][0][0](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Real_Aux[j][k][0](u,v)
												     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Real_Aux[j][k][1](u,v)
												     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Real_Aux[j][k][2](u,v) );
															  //^-core [0]	i.e., (i) is core		      ^-core [0]
									dcorr[1] += evj[u]*evj[v] * (  2.*(this->grad_evec_mm[i][k][0][1](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Real_Aux[j][k][0](u,v) 
												     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Real_Aux[j][k][1](u,v)
												     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Real_Aux[j][k][2](u,v) );

									dcorr[2] += evj[u]*evj[v] * (  2.*(this->grad_evec_mm[i][k][0][2](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Real_Aux[j][k][0](u,v)
												     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Real_Aux[j][k][1](u,v)
												     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Real_Aux[j][k][2](u,v) );
									
									// Reci
									dcorr[0] += evj[u]*evj[v] * (     (this->grad_evec_mm[i][k][0][0](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][0](0))*this->LPLP_H_Reci_Aux[j][k][0](u,v)	//  ss
												     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Reci_Aux[j][k][1](u,v)	// 2sx
												     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][2](u,v)	// 2sy
												     + 2.*(this->grad_evec_mm[i][k][0][0](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][3](u,v)	// 2sz
												     +    (this->grad_evec_mm[i][k][0][0](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][0](1))*this->LPLP_H_Reci_Aux[j][k][4](u,v)	//  xx
												     + 2.*(this->grad_evec_mm[i][k][0][0](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][5](u,v)	// 2xy
												     + 2.*(this->grad_evec_mm[i][k][0][0](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][6](u,v)	// 2xz
												     +    (this->grad_evec_mm[i][k][0][0](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][0](2))*this->LPLP_H_Reci_Aux[j][k][7](u,v)	//  yy
												     + 2.*(this->grad_evec_mm[i][k][0][0](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][8](u,v)	//  yz
												     +    (this->grad_evec_mm[i][k][0][0](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][0](3))*this->LPLP_H_Reci_Aux[j][k][9](u,v) );	//  zz

									dcorr[1] += evj[u]*evj[v] * (     (this->grad_evec_mm[i][k][0][1](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][1](0))*this->LPLP_H_Reci_Aux[j][k][0](u,v)	//  ss
												     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Reci_Aux[j][k][1](u,v)	// 2sx
												     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][2](u,v)	// 2sy
												     + 2.*(this->grad_evec_mm[i][k][0][1](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][3](u,v)	// 2sz
												     +    (this->grad_evec_mm[i][k][0][1](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][1](1))*this->LPLP_H_Reci_Aux[j][k][4](u,v)	//  xx
												     + 2.*(this->grad_evec_mm[i][k][0][1](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][5](u,v)	// 2xy
												     + 2.*(this->grad_evec_mm[i][k][0][1](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][6](u,v)	// 2xz
												     +    (this->grad_evec_mm[i][k][0][1](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][1](2))*this->LPLP_H_Reci_Aux[j][k][7](u,v)	//  yy
												     + 2.*(this->grad_evec_mm[i][k][0][1](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][8](u,v)	//  yz
												     +    (this->grad_evec_mm[i][k][0][1](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][1](3))*this->LPLP_H_Reci_Aux[j][k][9](u,v) );	//  zz

									dcorr[2] += evj[u]*evj[v] * (     (this->grad_evec_mm[i][k][0][2](0)*evk[0] + evk[0]*this->grad_evec_mm[i][k][0][2](0))*this->LPLP_H_Reci_Aux[j][k][0](u,v)	//  ss
												     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[1] + evk[0]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Reci_Aux[j][k][1](u,v)	// 2sx
												     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[2] + evk[0]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][2](u,v)	// 2sy
												     + 2.*(this->grad_evec_mm[i][k][0][2](0)*evk[3] + evk[0]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][3](u,v)	// 2sz
												     +    (this->grad_evec_mm[i][k][0][2](1)*evk[1] + evk[1]*this->grad_evec_mm[i][k][0][2](1))*this->LPLP_H_Reci_Aux[j][k][4](u,v)	//  xx
												     + 2.*(this->grad_evec_mm[i][k][0][2](1)*evk[2] + evk[1]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][5](u,v)	// 2xy
												     + 2.*(this->grad_evec_mm[i][k][0][2](1)*evk[3] + evk[1]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][6](u,v)	// 2xz
												     +    (this->grad_evec_mm[i][k][0][2](2)*evk[2] + evk[2]*this->grad_evec_mm[i][k][0][2](2))*this->LPLP_H_Reci_Aux[j][k][7](u,v)	//  yy
												     + 2.*(this->grad_evec_mm[i][k][0][2](2)*evk[3] + evk[2]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][8](u,v)	//  yz
												     +    (this->grad_evec_mm[i][k][0][2](3)*evk[3] + evk[3]*this->grad_evec_mm[i][k][0][2](3))*this->LPLP_H_Reci_Aux[j][k][9](u,v) );	//  zz
									}// v
								}// u
							}// end : if( k != j )
						}// end : if( C.AtomList[k]->type == "lone" )
					}// end : for(int k=0;k<C.NumberOfAtoms;k++)
				}// end : if( C.AtomList[j]->type == "lone" )
			}// end : for(int j=0;j<C.NumberOfAtoms;j++)
		}//end : if( C.AtomList[i]->type == "lone" )

		C.AtomList[i]->cart_gd(0) += dcorr[0];
		C.AtomList[i]->cart_gd(1) += dcorr[1];
		C.AtomList[i]->cart_gd(2) += dcorr[2];
		
		//printf("Check-2\n");
		//printf("%10.4lf\t%10.4lf\t%10.4lf\n",dcorr[0],dcorr[1],dcorr[2]);
	
		dcorr[0] = dcorr[1] = dcorr[2] = 0.;
	}// end : for(int i=0;i<C.NumberOfAtoms;i++)

	return;
}















////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

////	LonePair_StrainDerivative

////	////	////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////    ////

void Manager::StrainLonePairDerivativeReal( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;
        
        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" ) 
        {	
		// LonePair Core - Shell Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// LonePair Core - Shell Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" ) 
        {	
		// Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" ) 
        {
		// Shell Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
		// Shell Shell - LonePair Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart - TransVector;
		r_norm = Rij.norm();
		r_sqr  = r_norm*r_norm;

		intact = C.TO_EV*(0.5*Qi*Qj)*((-2./C.sigma/sqrt(M_PI))*(exp(-r_sqr/C.sigma/C.sigma)/r_norm)-(erfc(r_norm/C.sigma)/r_sqr))/r_norm;

		C.lattice_sd(0,0) += intact * Rij(0) * Rij(0);	C.lattice_sd(0,1) += intact * Rij(0) * Rij(1);	C.lattice_sd(0,2) += intact * Rij(0) * Rij(2);
								C.lattice_sd(1,1) += intact * Rij(1) * Rij(1);	C.lattice_sd(1,2) += intact * Rij(1) * Rij(2);
														C.lattice_sd(2,2) += intact * Rij(2) * Rij(2);
        }       
}       

void Manager::StrainLonePairDerivativeSelf( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double r_norm,r_sqr;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r - TransVector;
	double intact;

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )    // Handling Core - Core (i.e., charge charge interaction);
	{

	}
}

void Manager::StrainLonePairDerivativeReci( Cell& C, const int i, const int j, const Eigen::Vector3d& TransVector )
{
	double Qi,Qj;
	double g_norm = TransVector.norm();
	double g_sqr  = g_norm*g_norm;
	Eigen::Vector3d Rij;
        // TransVector(G) = 2pi h*u + 2pi k*v + 2pi l*w;
        // Rij            = Ai.r - Aj.r;
	double intact[4];

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "core" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "shel" ) 
        {
		// LonePair Core - Shell Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// LonePair Core - Shell Shell
		Qi = C.AtomList[i]->charge;
		Qj = static_cast<Shell*>(C.AtomList[j])->shel_charge;
		Rij = C.AtomList[i]->cart - static_cast<Shell*>(C.AtomList[j])->shel_cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "core" && C.AtomList[j]->type == "lone" ) 
        {
		// Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
        
	if( C.AtomList[i]->type == "shel" && C.AtomList[j]->type == "lone" ) 
        {
		// Shell Core - LonePair Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];

		// Shell Shell - LonePair Core
		Qi = static_cast<Shell*>(C.AtomList[i])->shel_charge;
		Qj = C.AtomList[j]->charge;
		Rij = static_cast<Shell*>(C.AtomList[i])->shel_cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       

        if( C.AtomList[i]->type == "lone" && C.AtomList[j]->type == "lone" )    // Handling Core - Core (i.e., charge charge interaction);
        {
		// LonePair Core - Core
		Qi = C.AtomList[i]->charge;
		Qj = C.AtomList[j]->charge;
		Rij = C.AtomList[i]->cart - C.AtomList[j]->cart;

		intact[0] = C.TO_EV*((2.*M_PI)/C.volume)*(Qi*Qj);
		intact[1] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr * -sin(TransVector.adjoint()*Rij);
		intact[2] = (-2.*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr/g_sqr*cos(TransVector.adjoint()*Rij)-0.5*exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*C.sigma*C.sigma*cos(TransVector.adjoint()*Rij));
		intact[3] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*sin(TransVector.adjoint()*Rij);
		intact[4] = exp(-0.25*C.sigma*C.sigma*g_sqr)/g_sqr*cos(TransVector.adjoint()*Rij);

		// Strain derivative (1) - w.r.t. r_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*intact[1] * TransVector(0) * Rij(0);	C.lattice_sd(0,1) += intact[0]*intact[1] * TransVector(0) * Rij(1);	C.lattice_sd(0,2) += intact[0]*intact[1] * TransVector(0) * Rij(2);
											C.lattice_sd(1,1) += intact[0]*intact[1] * TransVector(1) * Rij(1);	C.lattice_sd(1,2) += intact[0]*intact[1] * TransVector(1) * Rij(2);
																				C.lattice_sd(2,2) += intact[0]*intact[1] * TransVector(2) * Rij(2);
		// Strain derivative (2) - w.r.t. g_vector in the reciprocal space
		C.lattice_sd(0,0) += intact[0]*(intact[2]*TransVector(0)-intact[3]*Rij(0))*-TransVector(0);	C.lattice_sd(0,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(0);	C.lattice_sd(0,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(0);
														C.lattice_sd(1,1) += intact[0]*(intact[2]*TransVector(1)-intact[3]*Rij(1))*-TransVector(1);	C.lattice_sd(1,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(1);
																										C.lattice_sd(2,2) += intact[0]*(intact[2]*TransVector(2)-intact[3]*Rij(2))*-TransVector(2);
		// Strain derivative (3) - w.r.t cell volume in the reciprocal space
		C.lattice_sd(0,0) += -intact[0]*intact[4];	C.lattice_sd(1,1) += -intact[0]*intact[4];	C.lattice_sd(2,2) += -intact[0]*intact[4];
        }       
}
