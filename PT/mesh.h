#pragma once

#include <cuda_runtime.h>

#include <iostream>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "matrix.h"
#include "triangle.h"


struct					tri
{
	float3				vert[3];
	float3				norm[3];
	float2				uv[3];
};
struct					mesh_data
{
	std::vector<tri>	tri_data;
};


class mesh
{
private:
	static int						id_iter;
	static std::vector<mesh_data>	meshes;


	static std::vector<std::string>	str_split(const std::string &str, char delim)
	{
		std::stringstream			ss(str);
		std::string					line;
		std::vector<std::string>	vec;

		while (ss.peek() == delim)
			ss.ignore(1);

		while (std::getline(ss, line, delim))
			vec.push_back(line);

		return (vec);
	}

	static void						load_mesh_data(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, material_data mat_data, bound_type bvh_type, std::vector<shape*> &obj, std::vector<vol_data> &vol)
	{
		if (mesh_id < 0 || mesh_id >= mesh::meshes.size())
			return;
		

		mat4x4 t = mat4x4::create_translate_mat(pos);
		mat4x4 r = mat4x4::create_rot_mat(rotation);
		mat4x4 s = mat4x4::create_scale_mat(scale);
		mat4x4 world_mat = t * r * s;


		vol.push_back(vol_data());
		vol_data &_vol = vol[vol.size() - 1];
		_vol.shadow_visibility = mat_data.shadow_visibility;
		_vol.obj_type = MESH;
		_vol.bvh_type = bvh_type;
		_vol.range[0] = obj.size();


		int i = -1;
		while (++i < mesh::meshes[mesh_id].tri_data.size())
		{
			tri &tri_data = mesh::meshes[mesh_id].tri_data[i];

			float3 a = world_mat * tri_data.vert[0];
			float3 b = world_mat * tri_data.vert[1];
			float3 c = world_mat * tri_data.vert[2];
			

			mat4x4 normal_mat = mat4x4::transpose(mat4x4::inverse(world_mat));
			float3 a_n = normal_mat * tri_data.norm[0];
			float3 b_n = normal_mat * tri_data.norm[1];
			float3 c_n = normal_mat * tri_data.norm[2];

			float3 vert[3] = { a, b, c };
			float3 norm[3] = { a_n, b_n, c_n };
			float2 uv[3] = { tri_data.uv[0], tri_data.uv[1], tri_data.uv[2] };

			obj.push_back(new triangle(vert, norm, uv, mat_data));


			int f = -1;
			while (++f < 3)
			{
				if (vert[f].x < _vol.vol_min.x)
					_vol.vol_min.x = vert[f].x;
				if (vert[f].y < _vol.vol_min.y)
					_vol.vol_min.y = vert[f].y;
				if (vert[f].z < _vol.vol_min.z)
					_vol.vol_min.z = vert[f].z;

				if (vert[f].x > _vol.vol_max.x)
					_vol.vol_max.x = vert[f].x;
				if (vert[f].y > _vol.vol_max.y)
					_vol.vol_max.y = vert[f].y;
				if (vert[f].z > _vol.vol_max.z)
					_vol.vol_max.z = vert[f].z;
			}

			int p = -1;
			while (++p < BOUNDING_PLANES)
			{
				int f = -1;
				while (++f < 3)
				{
					float d = h_dot(BVH_NORMALS[p], vert[f]);//BVH_NORMALS[p].x * vert[f].x + BVH_NORMALS[p].y * vert[f].y + BVH_NORMALS[p].z * vert[f].z;
					_vol.plane_d_near[p] = fminf(d, _vol.plane_d_near[p]);
					_vol.plane_d_far[p] = fmaxf(d, _vol.plane_d_far[p]);
				}
			}
		}


		_vol.range[1] = obj.size() - 1;
	}

public:
	static int						load_mesh(const char *file_path)
	{
		std::vector<float3> vert;
		std::vector<float3> norm;
		std::vector<float2>	uv;

		std::ifstream is(file_path);
		if (!is)
			return (-1);

		mesh_data m_data;

		std::string line;
		while (std::getline(is, line, '\n'))
		{
			std::vector<std::string> split = str_split(line, ' ');
			if (!split.size())
				continue;

			if (split[0] == "v")
			{
				float3 v = make_float3(std::stof(split[1]), std::stof(split[2]), std::stof(split[3]));
				vert.push_back(v);
			}
			else if (split[0] == "vn")
			{
				float3 n = h_normalize(make_float3(std::stof(split[1]), std::stof(split[2]), std::stof(split[3])));
				norm.push_back(n);
			}
			else if (split[0] == "vt")
			{
				float2 t = make_float2(std::stof(split[1]), 1.0f - std::stof(split[2]));
				uv.push_back(t);
			}
			else if (split[0] == "f")
			{
				tri tri_data;

				int i = -1;
				while (++i < 3)
				{
					//		VERT/UV/NORM
					std::vector<std::string> vertex_info = str_split(split[i + 1], '/');

					tri_data.vert[i] = vert[std::stoi(vertex_info[0]) - 1];
					tri_data.uv[i] = uv[std::stoi(vertex_info[1]) - 1];
					tri_data.norm[i] = norm[std::stoi(vertex_info[2]) - 1];
				}

				m_data.tri_data.push_back(tri_data);
			}
		}

		int mesh_id = mesh::id_iter;
		mesh::meshes.push_back(m_data);
		mesh:id_iter++;

		return (mesh_id);
	}


	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, material_data mat_data, bound_type bvh_type, std::vector<shape*> &obj, std::vector<vol_data> &vol)
	{
		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, bvh_type, obj, vol);
	}
};
