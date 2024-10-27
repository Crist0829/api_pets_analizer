import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { extname } from 'path';

export const uploadInterceptor = FileInterceptor('file', {
  storage: diskStorage({
    destination: '../uploads',
    filename: (req, file, callback) => {
      const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
      const filename = `${uniqueSuffix}${extname(file.originalname)}`;
      callback(null, filename);
    },
  }),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB como tamaño máximo permitido
});