use crate::glm;

use crate::renderer::consts::*;
use glm::{Mat4, Vec2, Vec3};

#[derive(Clone)]
struct CamAutorotate {
    last_update: std::time::Instant,
    rate: Vec2,
}

#[derive(Clone)]
pub struct Camera {
    pub pos: Vec3,
    pub target: Vec3,
    pub aspect: f32,
    autorotate: Option<CamAutorotate>
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            pos: [0.0, 0.0, 3.0].into(),
            target: [0.0, 0.0, 0.0].into(),
            aspect,
            autorotate: None
        }
    }

    pub fn with_autorotate(mut self, rate: Vec2) -> Self {
        self.autorotate = Some(
            CamAutorotate { last_update: std::time::Instant::now(), rate }
        );
        self
    }

    pub fn set_autorotate(&mut self, rate: Vec2) -> &mut Self {
        self.autorotate = Some(
            CamAutorotate { last_update: std::time::Instant::now(), rate }
        );
        self
    }

    pub fn get_transform(&self) -> Mat4 {
        glm::look_at(&self.pos, &self.target, &UP)
    }

    pub fn get_quat(&self) -> glm::Quat {
        glm::quat_look_at(&(self.target - self.pos), &UP)
    }

    pub fn relative_pos(&self) -> Vec3 {
        self.pos - self.target
    }

    /// orbit camera around the target on the current ball the camera is on (centered at the
    /// target)
    ///
    /// will move radians equal to the magnitude of `delta` in the local direction of the
    /// components
    pub fn orbit_target(&mut self, delta: &Vec2) {
        let rel_pos = self.relative_pos();
        let horiz_normal = UP.cross(&rel_pos);
        let normal =
            glm::vec3(horiz_normal.x * delta.y, delta.x, horiz_normal.z * delta.y).normalize();
        let angle = delta.magnitude();
        let new_rel_pos = glm::rotate_vec3(&rel_pos, angle, &normal);
        // let new_rel_pos = glm::rotate_y_vec3(&self.relative_pos(), delta.x);
        self.pos = self.target + new_rel_pos;
    }
}

pub fn mouse_move(cam: &mut Camera, delta: &(f64, f64)) {
    let vdelta: Vec2;
    if let Some(ref mut auto) = cam.autorotate {
        let now = std::time::Instant::now();
        vdelta = auto.rate * now.duration_since(auto.last_update).as_secs_f32();
        auto.last_update = now;
    } else {
        vdelta = Into::<Vec2>::into([-delta.0 as f32, delta.1 as f32]) * 0.005;
    }
    cam.orbit_target(&vdelta);
}
