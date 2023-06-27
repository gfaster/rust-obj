use crate::glm;

use super::consts::*;
use glm::{Mat4, Vec2, Vec3};

pub struct Camera {
    pub pos: Vec3,
    pub target: Vec3,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: [0.0, 0.0, 3.0].into(),
            target: [0.0, 0.0, 0.0].into(),
        }
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

impl Default for Camera {
    fn default() -> Self {
        Self::new()
    }
}

pub fn mouse_move(cam: &mut Camera, delta: &(f32, f32)) {
    let vdelta: Vec2 = Into::<Vec2>::into([delta.1, delta.0]) * 0.005;
    cam.orbit_target(&vdelta);
}
