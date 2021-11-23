#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>
#include <cmath>
#include <vector>


namespace geometry {
    float DEFAULT_EPSILON = 0.000001f;
    float DEFAULT_EXTREME = 1000000.0f;

    struct point2d {
        float x;
        float y;
        point2d();
        point2d(float _x, float _y);
        point2d operator=(point2d pt);
        point2d operator()(float nx, float ny);
    };

    point2d::point2d() : x(0.0f), y(0.0f) {}

    point2d::point2d(float _x, float _y) : x(_x), y(_y) {}

    point2d point2d::operator=(point2d pt) {
        x = pt.x;
        y = pt.y;
        return *this;
    }

    point2d point2d::operator()(float nx, float ny) {
        x = nx;
        y = ny;
        return *this;
    }

    // Retorna la igualdad de dos números flotantes
    bool num_equal(float num1, float num2, float epsilon = DEFAULT_EPSILON) {
        return fabsf(num2 - num1) < epsilon;
    }

    // Calcula si un punto se encuentra en un segmento colinear
    bool point2d_on_segment2d(point2d seg_v0, point2d seg_v1, point2d p) {
        return ((p.x <= std::max(seg_v0.x, seg_v1.x) and (p.x >= std::min(seg_v0.x, seg_v1.x)) and
                (p.y <= std::max(seg_v0.y, seg_v1.y) and (p.y >= std::min(seg_v0.y, seg_v1.y)))));
    }

    // Calcula la interseccion de dos segmentos
    bool segments_intersection(point2d seg0_v0, point2d seg0_v1, point2d seg1_v0, point2d seg1_v1) {
        bool intersection = false;
        double r_x = seg0_v1.x - seg0_v0.x;
        double r_y = seg0_v1.y - seg0_v0.y;
        double s_x = seg1_v1.x - seg1_v0.x;
        double s_y = seg1_v1.y - seg1_v0.y;

        double det = (r_x * s_y) - (r_y * s_x);

        double t_0 = ((seg1_v0.x - seg0_v0.x) * s_y) - ((seg1_v0.y - seg0_v0.y) * s_x);
        double u_0 = ((seg1_v0.x - seg0_v0.x) * r_y) - ((seg1_v0.y - seg0_v0.y) * r_x);

        if (num_equal(det, 0.0)) {
            if (num_equal(t_0, 0.0) or num_equal(u_0, 0.0)) {
                if ((point2d_on_segment2d(seg0_v0, seg0_v1, seg1_v0)) or (point2d_on_segment2d(seg0_v0, seg0_v1, seg1_v1))) {
                    intersection = true;
                }
                else {
                    intersection = false;
                }
            }
            else {
                intersection = false;
            }
        }
        else {
            double t = t_0 / det;
            double u = u_0 / det;

            if (((t >= 0 and t <= 1) and (u >= 0 and u <= 1))) {
                intersection = true;
            }
            else {
                intersection = false;
            }
        }
        return intersection;
    }

    // Calcula si un punto se encuentra dentro de un poligono
    bool point2d_inside_polygon2d(std::vector<point2d> polygon, point2d p, float extreme = DEFAULT_EXTREME) {
        int polygon_size = polygon.size() - 1;
        point2d q(extreme, p.y);
        int counter = 0;
        for (int i = 0; i < polygon_size; i++) {
            if (segments_intersection(polygon[i], polygon[i+1], p, q)) {
                counter++;
            }
        }
        return counter & 1;
    }
}

#endif // GEOMETRY_H
