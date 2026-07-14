import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

const StudentPage = lazy(() => import('./pages/StudentPage'))
const TeacherPage = lazy(() => import('./pages/TeacherPage'))

export default function App() {
  return (
    <Suspense fallback={<div className="route-loading">正在加载工作台…</div>}>
      <Routes>
        <Route path="/" element={<Navigate to="/student" replace />} />
        <Route path="/student" element={<StudentPage />} />
        <Route path="/teacher" element={<TeacherPage />} />
        <Route path="*" element={<Navigate to="/student" replace />} />
      </Routes>
    </Suspense>
  )
}
