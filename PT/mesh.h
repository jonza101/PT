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

	static void						load_mesh_data(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, mesh_material_data mat_data, std::vector<shape*> &obj)
	{
		if (mesh_id < 0 || mesh_id >= mesh::meshes.size())
			return;

		mat4x4 t = mat4x4::create_translate_mat(pos);
		mat4x4 r = mat4x4::create_rot_mat(rotation);
		mat4x4 s = mat4x4::create_scale_mat(scale);
		mat4x4 world_mat = t * r * s;

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
		}
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


	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, float roughness, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, float roughness, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, float roughness, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, int roughness_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, int roughness_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, float roughness, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, int roughness_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, int roughness_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, float roughness, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, float roughness, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, float roughness, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, int roughness_id, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, int roughness_id, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, float roughness, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, int roughness_id, int normal_id, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, float roughness, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, float roughness, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, float roughness, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, int roughness_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, int roughness_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, float roughness, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, int albedo_id, float metalness, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo_id = albedo_id;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, float roughness, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness = roughness;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, int metalness_id, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness_id = metalness_id;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}

	static void						create_mesh(int mesh_id, const float3 &pos, const float3 &rotation, const float3 &scale, const float3 &albedo, float metalness, int roughness_id, int normal_id, int emissive_id, bool ef, float emissive, std::vector<shape*> &obj)
	{
		mesh_material_data mat_data;

		mat_data.albedo = albedo;
		mat_data.metalness = metalness;
		mat_data.roughness_id = roughness_id;
		mat_data.normal_id = normal_id;
		mat_data.emissive_id = emissive_id;
		mat_data.emissive = emissive;

		mesh::load_mesh_data(mesh_id, pos, rotation, scale, mat_data, obj);
	}
};
