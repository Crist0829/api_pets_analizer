import { IsOptional, IsString, IsEnum, Min } from 'class-validator'
import { PetsType } from '../entities/enums'

export class UpdatePetDto {
  @IsOptional()
  @IsString()
  name?: string

  @IsOptional()
  @IsString()
  color?: string

  @IsOptional()
  @IsEnum(PetsType, { message: 'type must be one of: cat, dog, other' })
  type?: PetsType

  @IsOptional()
  @Min(1)
  height?: number
}
