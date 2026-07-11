import { Navigate, Route, Routes } from 'react-router-dom'
import StudentPage from './pages/StudentPage'
import TeacherPage from './pages/TeacherPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/student" replace />} />
      <Route path="/student" element={<StudentPage />} />
      <Route path="/teacher" element={<TeacherPage />} />
      <Route path="*" element={<Navigate to="/student" replace />} />
    </Routes>
  )
}

