use glam::Mat4;
use macroquad::prelude::*;
use macroquad_profiler as profiler;
use std::f32::consts::PI;

fn draw_multiline_text(text: &str, x: f32, y: f32, font_size: f32, color: Color) {
    for (pos, text) in text.split('\n').enumerate() {
        draw_text(text, x, y + (pos as f32) * font_size, font_size, color);
    }
}

pub trait UniformStruct {
    fn uniforms(&self) -> Vec<(String, UniformType)>;
    fn set_uniforms(&self, material: Material);
}

pub struct MatWithInversion {
    matrix: Mat4,
    matrix_inverse: Mat4,

    name_matrix: String,
    name_matrix_inverse: String,
}

impl MatWithInversion {
    pub fn new(matrix: Mat4, name: &str) -> Self {
        Self {
            matrix_inverse: matrix.inverse(),
            matrix,

            name_matrix: name.to_owned(),
            name_matrix_inverse: format!("{}_inv", name),
        }
    }

    pub fn get(&self) -> &Mat4 {
        &self.matrix
    }

    pub fn get_inverse(&self) -> &Mat4 {
        &self.matrix_inverse
    }

    pub fn set(&mut self, matrix: Mat4) {
        self.matrix_inverse = matrix.inverse();
        self.matrix = matrix;
    }
}

impl UniformStruct for MatWithInversion {
    fn uniforms(&self) -> Vec<(String, UniformType)> {
        vec![
            (self.name_matrix.clone(), UniformType::Mat4),
            (self.name_matrix_inverse.clone(), UniformType::Mat4),
        ]
    }

    fn set_uniforms(&self, material: Material) {
        material.set_uniform(&self.name_matrix, self.matrix);
        material.set_uniform(&self.name_matrix_inverse, self.matrix_inverse);
    }
}

pub struct MatPortal {
    first: Mat4,
    first_teleport: Mat4,
    second: Mat4,
    second_teleport: Mat4,

    name_first: String,
    name_first_teleport: String,
    name_second: String,
    name_second_teleport: String,
}

impl MatPortal {
    pub fn new(first: Mat4, second: Mat4, name: &str) -> Self {
        Self {
            first_teleport: second.inverse() * first,
            second_teleport: first.inverse() * second,
            first,
            second,

            name_first: format!("{}_first", name),
            name_first_teleport: format!("{}_first_teleport", name),
            name_second: format!("{}_second", name),
            name_second_teleport: format!("{}_second_teleport", name),
        }
    }

    pub fn set(&mut self, first: Option<Mat4>, second: Option<Mat4>) {
        if let Some(new_first) = first {
            self.first = new_first;
        }
        if let Some(new_second) = second {
            self.second = new_second;
        }

        self.first_teleport = self.second.inverse() * self.first;
        self.second_teleport = self.first.inverse() * self.second;
    }
}

impl UniformStruct for MatPortal {
    fn uniforms(&self) -> Vec<(String, UniformType)> {
        vec![
            (self.name_first.clone(), UniformType::Mat4),
            (self.name_first_teleport.clone(), UniformType::Mat4),
            (self.name_second.clone(), UniformType::Mat4),
            (self.name_second_teleport.clone(), UniformType::Mat4),
        ]
    }

    fn set_uniforms(&self, material: Material) {
        material.set_uniform(&self.name_first, self.first);
        material.set_uniform(&self.name_first_teleport, self.first_teleport);
        material.set_uniform(&self.name_second, self.second);
        material.set_uniform(&self.name_second_teleport, self.second_teleport);
    }
}

struct Scene {
    mobius_portal: MatPortal,

    plane1: MatWithInversion,
    plane2: MatWithInversion,
    plane3: MatWithInversion,
    plane4: MatWithInversion,
    plane5: MatWithInversion,
    plane6: MatWithInversion,

    watermark: Texture2D,

    rotation_angle: f32,
}

impl Scene {
    fn textures(&self) -> Vec<String> {
        vec!["watermark".to_owned()]
    }

    fn first_portal() -> Mat4 {
        Mat4::from_rotation_x(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., 2.))
    }

    async fn new() -> Self {
        let watermark: Texture2D = load_texture("watermark.png").await;

        let first = Self::first_portal();
        let second =
            Mat4::from_rotation_x(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., -2.));

        let plane1 = Mat4::from_translation(Vec3::new(0., 0., 4.5));
        let plane2 = Mat4::from_translation(Vec3::new(0., 0., -4.5));
        let plane3 =
            Mat4::from_rotation_x(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., 4.5));
        let plane4 =
            Mat4::from_rotation_x(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., -4.5));
        let plane5 =
            Mat4::from_rotation_y(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., 4.5));
        let plane6 =
            Mat4::from_rotation_y(PI / 2.) * Mat4::from_translation(Vec3::new(0., 0., -4.5));

        Self {
            mobius_portal: MatPortal::new(first, second, "mobius_mat"),
            plane1: MatWithInversion::new(plane1, "plane1"),
            plane2: MatWithInversion::new(plane2, "plane2"),
            plane3: MatWithInversion::new(plane3, "plane3"),
            plane4: MatWithInversion::new(plane4, "plane4"),
            plane5: MatWithInversion::new(plane5, "plane5"),
            plane6: MatWithInversion::new(plane6, "plane6"),
            rotation_angle: 0.,
            watermark,
        }
    }

    fn process_mouse_and_keys(&mut self) -> bool {
        let mut is_something_changed = false;

        if is_key_down(KeyCode::A) {
            self.rotation_angle = clamp(self.rotation_angle + 1. / 180. * PI, 0., PI);
            self.mobius_portal.set(
                Some(Mat4::from_rotation_y(self.rotation_angle) * Self::first_portal()),
                None,
            );
            is_something_changed = true;
        }
        if is_key_down(KeyCode::B) {
            self.rotation_angle = clamp(self.rotation_angle - 1. / 180. * PI, 0., PI);
            self.mobius_portal.set(
                Some(Mat4::from_rotation_y(self.rotation_angle) * Self::first_portal()),
                None,
            );
            is_something_changed = true;
        }

        return is_something_changed;
    }
}

impl UniformStruct for Scene {
    fn uniforms(&self) -> Vec<(String, UniformType)> {
        let mut result = vec![];
        result.extend(self.mobius_portal.uniforms());
        result.extend(self.plane1.uniforms());
        result.extend(self.plane2.uniforms());
        result.extend(self.plane3.uniforms());
        result.extend(self.plane4.uniforms());
        result.extend(self.plane5.uniforms());
        result.extend(self.plane6.uniforms());
        result
    }

    fn set_uniforms(&self, material: Material) {
        self.mobius_portal.set_uniforms(material);
        self.plane1.set_uniforms(material);
        self.plane2.set_uniforms(material);
        self.plane3.set_uniforms(material);
        self.plane4.set_uniforms(material);
        self.plane5.set_uniforms(material);
        self.plane6.set_uniforms(material);
        material.set_texture("watermark", self.watermark);
    }
}

struct RotateAroundCam {
    alpha: f32,
    beta: f32,
    r: f32,
    previous_mouse: Vec2,
}

impl RotateAroundCam {
    const BETA_MIN: f32 = 0.01;
    const BETA_MAX: f32 = PI - 0.01;
    const MOUSE_SENSITIVITY: f32 = 1.2;
    const SCALE_FACTOR: f32 = 1.1;
    const VIEW_ANGLE: f32 = 80. / 180. * PI;

    fn new() -> Self {
        Self {
            alpha: PI,
            beta: PI / 2.,
            r: 3.5,
            previous_mouse: Vec2::default(),
        }
    }

    fn process_mouse_and_keys(&mut self) -> bool {
        let mut is_something_changed = false;

        let mouse_pos: Vec2 = mouse_position_local();

        if is_mouse_button_down(MouseButton::Left) {
            let dalpha = (mouse_pos.x - self.previous_mouse.x) * Self::MOUSE_SENSITIVITY;
            let dbeta = (mouse_pos.y - self.previous_mouse.y) * Self::MOUSE_SENSITIVITY;

            self.alpha += dalpha;
            self.beta = clamp(self.beta + dbeta, Self::BETA_MIN, Self::BETA_MAX);

            is_something_changed = true;
        }

        let wheel_value = mouse_wheel().1;
        if wheel_value > 0. {
            self.r *= 1.0 / Self::SCALE_FACTOR;
            is_something_changed = true;
        } else if wheel_value < 0. {
            self.r *= Self::SCALE_FACTOR;
            is_something_changed = true;
        }

        self.previous_mouse = mouse_pos;

        return is_something_changed;
    }

    fn get_matrix(&self) -> Mat4 {
        let pos = Vec3::new(
            -self.beta.sin() * self.alpha.cos(),
            self.beta.cos(),
            -self.beta.sin() * self.alpha.sin(),
        ) * self.r;
        let look_at = Vec3::new(0., 0., 0.);

        let h = (Self::VIEW_ANGLE / 2.).tan();

        let k = (look_at - pos).normalize();
        let i = k.cross(Vec3::new(0., 1., 0.)).normalize() * h;
        let j = k.cross(i).normalize() * h;

        Mat4::from_cols(
            Vec4::new(i.x, i.y, i.z, 0.),
            Vec4::new(j.x, j.y, j.z, 0.),
            Vec4::new(k.x, k.y, k.z, 0.),
            Vec4::new(pos.x, pos.y, pos.z, 1.),
        )
    }
}

impl UniformStruct for RotateAroundCam {
    fn uniforms(&self) -> Vec<(String, UniformType)> {
        vec![("camera".to_owned(), UniformType::Mat4)]
    }

    fn set_uniforms(&self, material: Material) {
        material.set_uniform("camera", self.get_matrix());
    }
}

struct Window {
    add_gray_after_teleportation: f32,
    teleport_light: bool,
    show_help: bool,
    show_profiler: bool,

    scene: Scene,
    cam: RotateAroundCam,
}

impl Window {
    async fn new() -> Self {
        Window {
            add_gray_after_teleportation: 1.0,
            teleport_light: true,
            show_help: true,
            show_profiler: false,

            scene: Scene::new().await,
            cam: RotateAroundCam::new(),
        }
    }

    fn process_mouse_and_keys(&mut self) -> bool {
        let mut is_something_changed = false;

        if is_key_pressed(KeyCode::H) {
            self.show_help = !self.show_help;
            is_something_changed = true;
        }
        if is_key_pressed(KeyCode::T) {
            self.teleport_light = !self.teleport_light;
            is_something_changed = true;
        }
        if is_key_pressed(KeyCode::P) {
            self.show_profiler = !self.show_profiler;
            is_something_changed = true;
        }
        if is_key_down(KeyCode::X) {
            self.add_gray_after_teleportation =
                clamp(self.add_gray_after_teleportation - 0.01, 0., 1.);
            is_something_changed = true;
        }
        if is_key_down(KeyCode::Y) {
            self.add_gray_after_teleportation =
                clamp(self.add_gray_after_teleportation + 0.01, 0., 1.);
            is_something_changed = true;
        }
        is_something_changed |= self.scene.process_mouse_and_keys();
        is_something_changed |= self.cam.process_mouse_and_keys();

        return is_something_changed;
    }

    fn draw(&self, material: Material) {
        gl_use_material(material);
        draw_rectangle(0., 0., screen_width(), screen_height(), WHITE);
        gl_use_default_material();

        if self.show_help {
            draw_multiline_text(
                "h - hide this message\nt - enable texture on Mobius strip\na/b - rotate blue portal\nx/y - make teleported rays darker\np - enable profiler",
                5.0,
                15.0,
                20.0,
                BLACK,
            );
        }
        if self.show_profiler {
            set_default_camera();
            profiler::profiler(profiler::ProfilerParams {
                fps_counter_pos: vec2(10.0, 10.0),
            });
        }
    }
}

impl UniformStruct for Window {
    fn uniforms(&self) -> Vec<(String, UniformType)> {
        let mut result = vec![
            ("resolution".to_owned(), UniformType::Float2),
            (
                "add_gray_after_teleportation".to_owned(),
                UniformType::Float1,
            ),
            ("teleport_light".to_owned(), UniformType::Int1),
        ];
        result.extend(self.scene.uniforms());
        result.extend(self.cam.uniforms());
        result
    }

    fn set_uniforms(&self, material: Material) {
        material.set_uniform("resolution", (screen_width(), screen_height()));
        material.set_uniform(
            "add_gray_after_teleportation",
            self.add_gray_after_teleportation,
        );
        material.set_uniform("teleport_light", self.teleport_light as i32);

        self.scene.set_uniforms(material);
        self.cam.set_uniforms(material);
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Portal visualization".to_owned(),
        high_dpi: true,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut window = Window::new().await;

    let lens_material = load_material(
        VERTEX_SHADER,
        FRAGMENT_SHADER,
        MaterialParams {
            uniforms: window.uniforms(),
            textures: window.scene.textures(),
            ..Default::default()
        },
    )
    .unwrap_or_else(|err| {
        if let miniquad::graphics::ShaderError::CompilationError { error_message, .. } = err {
            println!("Fragment shader compilation error:\n{}", error_message);
        } else {
            println!("Other material error:\n{:#?}", err);
        }
        std::process::exit(1)
    });

    let mut texture = load_texture_from_image(&get_screen_data());
    let mut w = screen_width();
    let mut h = screen_height();
    let mut image_size_changed = true;

    loop {
        clear_background(BLACK);

        if (screen_width() - w).abs() > 0.5 {
            w = screen_width();
            image_size_changed = true;
        }
        if (screen_height() - h).abs() > 0.5 {
            h = screen_height();
            image_size_changed = true;
        }
        if image_size_changed {
            texture = load_texture_from_image(&get_screen_data());
        }

        if window.process_mouse_and_keys() || image_size_changed {
            window.set_uniforms(lens_material);
            window.draw(lens_material);
            set_default_camera();
            texture.grab_screen();
            image_size_changed = false;
        } else {
            draw_texture_ex(texture, 0., 0., WHITE, DrawTextureParams {
                dest_size: Some(Vec2::new(screen_width(), screen_height())),
                flip_y: true,
                ..Default::default()
            });
        }

        next_frame().await;
    }
}

const FRAGMENT_SHADER: &'static str = include_str!("frag.glsl");

const VERTEX_SHADER: &'static str = "#version 100
attribute vec3 position;
attribute vec2 texcoord;

varying lowp vec2 uv;
varying lowp vec2 uv_screen;

uniform mat4 Model;
uniform mat4 Projection;

uniform vec2 Center;
uniform vec2 resolution;

void main() {
    vec4 res = Projection * Model * vec4(position, 1);

    uv_screen = (position.xy - resolution/2.) / min(resolution.x, resolution.y) * 2.;
    uv_screen.y *= -1.;
    uv = texcoord;

    gl_Position = res;
}
";
