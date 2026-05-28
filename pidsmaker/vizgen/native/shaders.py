import vispy.scene
from vispy.scene.visuals import create_visual_node

base_vertex_shader = """
uniform float u_antialias;
uniform float u_px_scale;
uniform bool u_scaling;
uniform bool u_spherical;
uniform float u_canvas_size_min;
uniform float u_canvas_size_max;

attribute vec3 a_position;
attribute vec4 a_fg_color;
attribute vec4 a_bg_color;
attribute float a_edgewidth;
attribute float a_size;
attribute float a_symbol;

varying vec4 v_fg_color;
varying vec4 v_bg_color;
varying float v_edgewidth;
varying float v_total_size;
varying float v_depth_middle;
varying float v_alias_ratio;
varying float v_symbol;

float big_float = 1e10;

// --- TEMPORAL UNIFORMS ---
uniform float u_time;
attribute float a_tw_start;
attribute float a_tw_end;

void main (void) {
    v_fg_color  = a_fg_color;
    v_bg_color  = a_bg_color;
    v_symbol = a_symbol + 0.5;

    float current_size = a_size;
    
    // --- TEMPORAL GLSL MATH ---
    if (u_time >= 0.0) {
        float age = u_time - a_tw_start;
        // Node hasn't appeared yet, or it has permanently died
        if (u_time < a_tw_start || u_time >= a_tw_end) {
            v_bg_color.a = 0.0;
            v_fg_color.a = 0.0;
            current_size = 0.0; // Shrink to 0 to skip rasterization completely
        } else {
            // Thermal Heatmap (Nodes glow hot when they first appear)
            float blend = clamp(1.0 - (age / 1.5), 0.0, 1.0);
            vec3 hot_color = vec3(1.0, 1.0, 0.8);
            v_bg_color.rgb = mix(v_bg_color.rgb, hot_color, blend);
            v_fg_color.rgb = mix(v_fg_color.rgb, hot_color, blend);
            
            // Age Ghosting (Nodes fade to 0.5 opacity as they get older)
            float alphas = clamp(1.0 - (age * 0.3), 0.50, 1.0);
            v_bg_color.a *= alphas;
            v_fg_color.a *= alphas;
        }
    }

    $setup_texcoord();

    vec4 pos = vec4(a_position, 1);
    vec4 fb_pos = $visual_to_framebuffer(pos);
    vec4 x;
    vec4 size_vec;

    if (u_scaling) {
        pos = $framebuffer_to_scene_or_visual(fb_pos);
        x = $framebuffer_to_scene_or_visual(fb_pos + vec4(big_float, 0, 0, 0));
        x = (x - pos);
        size_vec = $scene_or_visual_to_framebuffer(pos + normalize(x) * current_size);
        $v_size = size_vec.x / size_vec.w - fb_pos.x / fb_pos.w;
        v_edgewidth = ($v_size / current_size) * a_edgewidth;
    }
    else {
        $v_size = current_size * u_px_scale;
        v_edgewidth = a_edgewidth * u_px_scale;
    }

    float original_size = $v_size;
    if (u_canvas_size_min >= 0.0) {
        $v_size = max($v_size, u_canvas_size_min);
    }
    if (u_canvas_size_max >= 0.0) {
        $v_size = min($v_size, u_canvas_size_max);
    }
    if ($v_size != original_size) {
        v_edgewidth = v_edgewidth * ($v_size / original_size);
        if (u_canvas_size_min >= 0.0) {
            v_edgewidth = max(v_edgewidth, u_canvas_size_min * 0.5);
        }
        v_edgewidth = min(v_edgewidth, $v_size * 0.5);
    }

    float total_size = $v_size + 4. * (v_edgewidth + 1.5 * u_antialias);
    v_total_size = total_size;

    vec4 final_fb_pos = $apply_offset(fb_pos, total_size);
    gl_Position = $framebuffer_to_render(final_fb_pos);
    gl_PointSize = total_size;

    if (u_spherical == true) {
        vec4 z = $framebuffer_to_scene_or_visual(fb_pos + vec4(0, 0, big_float, 0));
        z = (z - pos);
        vec4 depth_z_vec = $scene_or_visual_to_framebuffer(pos + normalize(z) * current_size / 2);
        v_depth_middle = depth_z_vec.z / depth_z_vec.w - fb_pos.z / fb_pos.w;
        v_alias_ratio = total_size / $v_size;
    }
}
"""

class TemporalMarkersVisual(vispy.visuals.MarkersVisual):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shaders['vertex'] = base_vertex_shader
        self.shared_program.vert = self._shaders['vertex']

TemporalMarkers = create_visual_node(TemporalMarkersVisual)
