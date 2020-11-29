#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <iostream>


class		mat4x4
{
private:

	float			m[4][4] = { 0 };


	static mat4x4		create_rot_mat_x(float angle_rad)
	{
		mat4x4 mat = mat4x4::create_identity_mat();

		mat[1][1] = cosf(angle_rad);
		mat[1][2] = -sinf(angle_rad);
		mat[2][1] = sinf(angle_rad);
		mat[2][2] = cosf(angle_rad);

		return (mat);
	}

	static mat4x4		create_rot_mat_y(float angle_rad)
	{
		mat4x4 mat = mat4x4::create_identity_mat();

		mat[0][0] = cosf(angle_rad);
		mat[0][2] = sinf(angle_rad);
		mat[2][0] = -sinf(angle_rad);
		mat[2][2] = cosf(angle_rad);

		return (mat);
	}

	static mat4x4		create_rot_mat_z(float angle_rad)
	{
		mat4x4 mat = mat4x4::create_identity_mat();

		mat[0][0] = cosf(angle_rad);
		mat[0][1] = -sinf(angle_rad);
		mat[1][0] = sinf(angle_rad);
		mat[1][1] = cosf(angle_rad);

		return (mat);
	}

public:
	static mat4x4		create_identity_mat()
	{
		mat4x4 mat;

		mat[0][0] = 1.0f;
		mat[1][1] = 1.0f;
		mat[2][2] = 1.0f;
		mat[3][3] = 1.0f;

		return (mat);
	}

	static mat4x4		inverse(const mat4x4 &mat)
	{
		mat4x4 m;

		float det =		mat[0][0] * mat[1][1] * mat[2][2] * mat[3][3]	+ mat[0][0] * mat[1][2] * mat[2][3] * mat[3][1]		+ mat[0][0] * mat[1][3] * mat[2][1] * mat[3][2]
					+	mat[0][1] * mat[1][0] * mat[2][3] * mat[3][2]	+ mat[0][1] * mat[1][2] * mat[2][0] * mat[3][3]		+ mat[0][1] * mat[1][3] * mat[2][2] * mat[3][0]
					+	mat[0][2] * mat[1][0] * mat[2][1] * mat[3][3]	+ mat[0][2] * mat[1][1] * mat[2][3] * mat[3][0]		+ mat[0][2] * mat[1][3] * mat[2][0] * mat[3][1]
					+	mat[0][3] * mat[1][0] * mat[2][2] * mat[3][1]	+ mat[0][3] * mat[1][1] * mat[2][0] * mat[3][2]		+ mat[0][3] * mat[1][2] * mat[2][1] * mat[3][0]
					-	mat[0][0] * mat[1][1] * mat[2][3] * mat[3][2]	- mat[0][0] * mat[1][2] * mat[2][1] * mat[3][3]		- mat[0][0] * mat[1][3] * mat[2][2] * mat[3][1]
					-	mat[0][1] * mat[1][0] * mat[2][2] * mat[3][3]	- mat[0][1] * mat[1][2] * mat[2][3] * mat[3][0]		- mat[0][1] * mat[1][3] * mat[2][0] * mat[3][2]
					-	mat[0][2] * mat[1][0] * mat[2][3] * mat[3][1]	- mat[0][2] * mat[1][1] * mat[2][0] * mat[3][3]		- mat[0][2] * mat[1][3] * mat[2][1] * mat[3][0]
					-	mat[0][3] * mat[1][0] * mat[2][1] * mat[3][2]	- mat[0][3] * mat[1][1] * mat[2][2] * mat[3][0]		- mat[0][3] * mat[1][2] * mat[2][0] * mat[3][1];
		if (det == 0.0f)
			std::cout << "zero\n";
		det = 1.0f / (float)det;


		m[0][0] = mat[1][1] * mat[2][2] * mat[3][3] + mat[1][2] * mat[2][3] * mat[3][1] + mat[1][3] * mat[2][1] * mat[3][2] - mat[1][1] * mat[2][3] * mat[3][2] - mat[1][2] * mat[2][1] * mat[3][3] - mat[1][3] * mat[2][2] * mat[3][1];
		m[0][1] = mat[0][1] * mat[2][3] * mat[3][2] + mat[0][2] * mat[2][1] * mat[3][3] + mat[0][3] * mat[2][2] * mat[3][1] - mat[0][1] * mat[2][2] * mat[3][3] - mat[0][2] * mat[2][3] * mat[3][1] - mat[0][3] * mat[2][1] * mat[3][2];
		m[0][2] = mat[0][1] * mat[1][2] * mat[3][3] + mat[0][2] * mat[1][3] * mat[3][1] + mat[0][3] * mat[1][1] * mat[3][2] - mat[0][1] * mat[1][3] * mat[3][2] - mat[0][2] * mat[1][1] * mat[3][3] - mat[0][3] * mat[1][2] * mat[3][1];
		m[0][3] = mat[0][1] * mat[1][3] * mat[2][2] + mat[0][2] * mat[1][1] * mat[2][3] + mat[0][3] * mat[1][2] * mat[2][1] - mat[0][1] * mat[1][2] * mat[2][3] - mat[0][2] * mat[1][3] * mat[2][1] - mat[0][3] * mat[1][1] * mat[2][2];
		m[1][0] = mat[1][0] * mat[2][3] * mat[3][2] + mat[1][2] * mat[2][0] * mat[3][3] + mat[1][3] * mat[2][2] * mat[3][0] - mat[1][0] * mat[2][2] * mat[3][3] - mat[1][2] * mat[2][3] * mat[3][0] - mat[1][3] * mat[2][0] * mat[3][2];
		m[1][1] = mat[0][0] * mat[2][2] * mat[3][3] + mat[0][2] * mat[2][3] * mat[3][0] + mat[0][3] * mat[2][0] * mat[3][2] - mat[0][0] * mat[2][3] * mat[3][2] - mat[0][2] * mat[2][0] * mat[3][3] - mat[0][3] * mat[2][2] * mat[3][0];
		m[1][2] = mat[0][0] * mat[1][3] * mat[3][2] + mat[0][2] * mat[1][0] * mat[3][3] + mat[0][3] * mat[1][2] * mat[3][0] - mat[0][0] * mat[1][2] * mat[3][3] - mat[0][2] * mat[1][3] * mat[3][0] - mat[0][3] * mat[1][0] * mat[3][2];
		m[1][3] = mat[0][0] * mat[1][2] * mat[2][3] + mat[0][2] * mat[1][3] * mat[2][0] + mat[0][3] * mat[1][0] * mat[2][2] - mat[0][0] * mat[1][3] * mat[2][2] - mat[0][2] * mat[1][0] * mat[2][3] - mat[0][3] * mat[1][2] * mat[2][0];
		m[2][0] = mat[1][0] * mat[2][1] * mat[3][3] + mat[1][1] * mat[2][3] * mat[3][0] + mat[1][3] * mat[2][0] * mat[3][1] - mat[1][0] * mat[2][3] * mat[3][1] - mat[1][1] * mat[2][0] * mat[3][3] - mat[1][3] * mat[2][1] * mat[3][0];
		m[2][1] = mat[0][0] * mat[2][3] * mat[3][1] + mat[0][1] * mat[2][0] * mat[3][3] + mat[0][3] * mat[2][1] * mat[3][0] - mat[0][0] * mat[2][1] * mat[3][3] - mat[0][1] * mat[2][3] * mat[3][0] - mat[0][3] * mat[2][0] * mat[3][1];
		m[2][2] = mat[0][0] * mat[1][1] * mat[3][3] + mat[0][1] * mat[1][3] * mat[3][0] + mat[0][3] * mat[1][0] * mat[3][1] - mat[0][0] * mat[1][3] * mat[3][1] - mat[0][1] * mat[1][0] * mat[3][3] - mat[0][3] * mat[1][1] * mat[3][0];
		m[2][3] = mat[0][0] * mat[1][3] * mat[2][1] + mat[0][1] * mat[1][0] * mat[2][3] + mat[0][3] * mat[1][1] * mat[2][0] - mat[0][0] * mat[1][1] * mat[2][3] - mat[0][1] * mat[1][3] * mat[2][0] - mat[0][3] * mat[1][0] * mat[2][1];
		m[3][0] = mat[1][0] * mat[2][2] * mat[3][1] + mat[1][1] * mat[2][0] * mat[3][2] + mat[1][2] * mat[2][1] * mat[3][0] - mat[1][0] * mat[2][1] * mat[3][2] - mat[1][1] * mat[2][2] * mat[3][0] - mat[1][2] * mat[2][0] * mat[3][1];
		m[3][1] = mat[0][0] * mat[2][1] * mat[3][2] + mat[0][1] * mat[2][2] * mat[3][0] + mat[0][2] * mat[2][0] * mat[3][1] - mat[0][0] * mat[2][2] * mat[3][1] - mat[0][1] * mat[2][0] * mat[3][2] - mat[0][2] * mat[2][1] * mat[3][0];
		m[3][2] = mat[0][0] * mat[1][2] * mat[3][1] + mat[0][1] * mat[1][0] * mat[3][2] + mat[0][2] * mat[1][1] * mat[3][0] - mat[0][0] * mat[1][1] * mat[3][2] - mat[0][1] * mat[1][2] * mat[3][0] - mat[0][2] * mat[1][0] * mat[3][1];
		m[3][3] = mat[0][0] * mat[1][1] * mat[2][2] + mat[0][1] * mat[1][2] * mat[2][0] + mat[0][2] * mat[1][0] * mat[2][1] - mat[0][0] * mat[1][2] * mat[2][1] - mat[0][1] * mat[1][0] * mat[2][2] - mat[0][2] * mat[1][1] * mat[2][0];


		int i = -1;
		while (++i < 4)
		{
			int j = -1;
			while (++j < 4)
			{
				m[i][j] *= det;
			}
		}

		return (m);

	}

	static mat4x4		transpose(const mat4x4 &mat)
	{
		mat4x4 m;

		int i = -1;
		while (++i < 4)
		{
			int j = -1;
			while (++j < 4)
			{
				m[j][i] = mat[i][j];
			}
		}

		return (m);
	}


	static mat4x4		create_rot_mat(const float3 &rotation)			//	Z * Y * X
	{
		mat4x4 mat;

		mat4x4 x = mat4x4::create_rot_mat_x(rotation.x);
		mat4x4 y = mat4x4::create_rot_mat_y(rotation.y);
		mat4x4 z = mat4x4::create_rot_mat_z(rotation.z);
		mat = z * y * x;

		return (mat);
	}

	static mat4x4		create_scale_mat(const float3 &scale)
	{
		mat4x4 mat = mat4x4::create_identity_mat();

		mat[0][0] = scale.x;
		mat[1][1] = scale.y;
		mat[2][2] = scale.z;

		return (mat);
	}

	static mat4x4		create_translate_mat(const float3 &trans)
	{
		mat4x4 mat = mat4x4::create_identity_mat();

		mat[0][3] = trans.x;
		mat[1][3] = trans.y;
		mat[2][3] = trans.z;

		return (mat);
	}


	float3			operator * (const float3 &v)
	{
		float3 vec;

		vec.x = this->m[0][0] * v.x + this->m[0][1] * v.y + this->m[0][2] * v.z + this->m[0][3];
		vec.y = this->m[1][0] * v.x + this->m[1][1] * v.y + this->m[1][2] * v.z + this->m[1][3];
		vec.z = this->m[2][0] * v.x + this->m[2][1] * v.y + this->m[2][2] * v.z + this->m[2][3];

		return (vec);
	}

	mat4x4			operator * (const mat4x4 &m)
	{
		mat4x4 mat;

		int i = -1;
		while (++i < 4)
		{
			int j = -1;
			while (++j < 4)
				mat[i][j] = this->m[i][0] * m[0][j] + this->m[i][1] * m[1][j] + this->m[i][2] * m[2][j] + this->m[i][3] * m[3][j];
		}

		return (mat);
	}


	float const		*operator [] (size_t idx) const
	{
		return (this->m[idx]);
	}

	float			*operator [] (size_t idx)
	{
		return (this->m[idx]);
	}
};
