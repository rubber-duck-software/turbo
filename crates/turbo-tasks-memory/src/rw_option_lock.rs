use std::ops::{Deref, DerefMut};

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub struct RwOptionLock<T> {
    inner: RwLock<Option<Box<T>>>,
}

impl<T> RwOptionLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: RwLock::new(Some(box value)),
        }
    }

    pub fn read(&self) -> Option<ReadGuard<'_, T>> {
        let inner = self.inner.read();
        if inner.is_none() {
            None
        } else {
            Some(ReadGuard { inner })
        }
    }

    pub fn write_or_init(&self, init: impl FnOnce() -> T) -> WriteGuard<'_, T> {
        let mut inner = self.inner.write();
        if inner.is_none() {
            *inner = Some(box init());
        }
        WriteGuard { inner }
    }

    pub fn write(&self) -> Option<WriteGuard<'_, T>> {
        let inner = self.inner.write();
        if inner.is_none() {
            None
        } else {
            Some(WriteGuard { inner })
        }
    }

    pub fn try_read(&self) -> Option<ReadGuard<'_, T>> {
        if let Some(inner) = self.inner.try_read() {
            if inner.is_some() {
                return Some(ReadGuard { inner });
            }
        }
        None
    }
}

pub struct ReadGuard<'a, T> {
    inner: RwLockReadGuard<'a, Option<Box<T>>>,
}

impl<'a, T> Deref for ReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

pub struct WriteGuard<'a, T> {
    inner: RwLockWriteGuard<'a, Option<Box<T>>>,
}

impl<'a, T> Deref for WriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for WriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}
